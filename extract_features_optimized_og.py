import raw_dataset as dataset
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import gc
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")

# Setup argparse
parser = argparse.ArgumentParser(description='Feature extraction for anti-spoofing')
parser.add_argument('--dataset', type=str, required=True, 
                    choices=['codecfake', 'asvspoof2019', 'asvspoof5'],
                    help='Which dataset to process')
parser.add_argument('--dataset_path', type=str, required=True, 
                    help='Path to the dataset folder')
parser.add_argument('--output_path', type=str, required=True, 
                    help='Path to the output directory')

parser.add_argument('--model_type', type=str, required=True,help='Which Wav2Vec2 features to extract')

parser.add_argument('--gpu', type=str, default='0', help='GPU index')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for processing')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of dataloader workers')
parser.add_argument('--save_workers', type=int, default=8,
                    help='Number of workers for parallel file saving')
parser.add_argument('--use_fp16', action='store_true', default=True,
                    help='Use mixed precision for faster processing')
parser.add_argument('--skip_existing', default=True,
                    help='Skip already processed files')

args = parser.parse_args()

# Setup GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cuda = torch.cuda.is_available()
print('='*80)
print(f"CUDA available: {cuda}")
print(f"Configuration:")
print(f"  Batch size: {args.batch_size}")
print(f"  Dataloader workers: {args.num_workers}")
print(f"  Save workers: {args.save_workers}")
print(f"  Mixed precision (FP16): {args.use_fp16}")
print(f"  Skip existing: {args.skip_existing}")
print('='*80)

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def deterministic_crop_or_pad(wav, target_length=16000, crop_idx=0, num_crops=3):
    """
    Crop o pad deterministico per consistency
    """
    waveform = wav.squeeze(0) if wav.dim() > 1 else wav
    waveform_len = waveform.shape[0]
    
    if waveform_len >= target_length:
        if num_crops == 1:
            # Centro per single crop
            start_idx = (waveform_len - target_length) // 2
        else:
            # Multi-crop: inizio, centro, fine
            if crop_idx == 0:
                start_idx = 0
            elif crop_idx == 1:
                start_idx = (waveform_len - target_length) // 2
            else:
                start_idx = waveform_len - target_length
        
        return waveform[start_idx:start_idx + target_length]
    else:
        # Padding con ripetizione
        num_repeats = (target_length + waveform_len - 1) // waveform_len
        padded_waveform = waveform.repeat(num_repeats)[:target_length]
        return padded_waveform

class FastAudioDataset(Dataset):
    """Dataset ottimizzato per processing veloce"""
    def __init__(self, raw_dataset, processor, num_crops=1, dataset_name="codecfake", part="train"):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.num_crops = num_crops
        self.dataset_name = dataset_name
        self.part = part
        self.total_items = len(raw_dataset) * num_crops
        
    def __len__(self):
        return self.total_items
    
    def __getitem__(self, idx):
        # Calcola indici relativi al dataset corrente
        raw_idx = idx // self.num_crops
        crop_idx = idx % self.num_crops
        
        try:
            # Get data based on dataset type
            if self.dataset_name == "asvspoof2019":
                waveform, filename, label = self.raw_dataset[raw_idx]
                tag = None
            else:  # codecfake
                waveform, filename, label = self.raw_dataset[raw_idx]
                tag = None
            
            # Skip silenzioso
            if waveform.sum() == 0:
                return None
            
            # Crop/pad con target length appropriato
            cropped = deterministic_crop_or_pad(
                waveform,
                target_length=16000,  # 1 secondi a 16kHz
                crop_idx=crop_idx,
                num_crops=self.num_crops
            )
            
            # Processa per Wav2Vec2
            input_values = self.processor(
                cropped.numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.squeeze(0)
            
            return {
                'input_values': input_values,
                'filename': filename,
                'label': label,
                'raw_idx': raw_idx,
                'crop_idx': crop_idx,
                'tag': tag if self.dataset_name == "asvspoof2019" else None
            }
            
        except Exception as e:
            print(f"Error loading sample {raw_idx}: {e}")
            return None

def collate_fn_fast(batch):
    """Collate function ottimizzata"""
    # Filtra None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    input_values = torch.stack([item['input_values'] for item in batch])
    filenames = [item['filename'] for item in batch]
    labels = [item['label'] for item in batch]
    raw_indices = [item['raw_idx'] for item in batch]
    crop_indices = [item['crop_idx'] for item in batch]
    tags = [item.get('tag') for item in batch]
    
    return {
        'input_values': input_values,
        'filenames': filenames,
        'labels': labels,
        'raw_indices': raw_indices,
        'crop_indices': crop_indices,
        'tags': tags
    }


from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import torch

# LA SICURA
'''def save_features_batch(features_list, target_dir, dataset_name="codecfake"):
    """Salva features in parallelo, un file .pt per sample"""
    os.makedirs(target_dir, exist_ok=True)

    def save_single(item):
        features, filename, label, raw_idx, crop_idx, tag = item

        crop_idx = int(crop_idx)
    

        # Normalizza label: gestisce sia stringhe sia interi
        if isinstance(label, str):
            label_str = label
            label_map = {
                "bonafide": 0,
                "spoof": 1,
                "real": 0,
                "fake": 1
            }
            if label_str not in label_map:
                raise ValueError(f"Label sconosciuta: {label_str}")
            label_num = label_map[label_str]
        else:
            label_num = int(label)
            label_str = "spoof" if label_num == 1 else "bonafide"

        # Naming del file
        # Se vuoi mantenere naming numerico, usa label_num
        # Se vuoi naming testuale, usa label_str
        if crop_idx > 0:
            output_file = os.path.join(
                target_dir,
                f"{filename}_crop_{crop_idx}_label_{label_num}.pt"
            )
        else:
            output_file = os.path.join(
                target_dir,
                f"{filename}_label_{label_num}.pt"
            )

        if args.skip_existing and os.path.exists(output_file):
            return ("skipped", output_file)

        # Qui features è già su CPU e pronto
        torch.save(features, output_file)
        return ("saved", output_file)

    results = {"saved": 0, "skipped": 0, "failed": 0}
    futures = []

    with ThreadPoolExecutor(max_workers=args.save_workers) as executor:
        for item in features_list:
            futures.append(executor.submit(save_single, item))

        for future in as_completed(futures):
            try:
                status, path = future.result()
                results[status] += 1
            except Exception as e:
                results["failed"] += 1
                print(f"Errore nel salvataggio: {e}")

    return results'''

# Versione senza blocco
def save_features_batch(features_list, target_dir, dataset_name="codecfake", existing_files = None):
    """Salva features in parallelo, un file .pt per sample, senza bloccare il flusso se un file fallisce."""
    os.makedirs(target_dir, exist_ok=True)
    if existing_files is None:
        existing_files = set()

    def save_single(item):
        try:
            features, filename, label, raw_idx, crop_idx, tag = item

            crop_idx = int(crop_idx)

            # Normalizza label
            if isinstance(label, str):
                label_str = label
                label_map = {
                    "bonafide": 0,
                    "spoof": 1,
                    "real": 0,
                    "fake": 1
                }
                if label_str not in label_map:
                    return ("failed", None, f"Label sconosciuta: {label_str}")
                label_num = label_map[label_str]
            else:
                label_num = int(label)

            # Naming
            if crop_idx > 0:
                output_file = os.path.join(
                    target_dir,
                    f"{filename}_crop_{crop_idx}_label_{label_num}.pt"
                )
            else:
                output_file = os.path.join(
                    target_dir,
                    f"{filename}_label_{label_num}.pt"
                )

            '''# Skip se già esiste
            if args.skip_existing and os.path.exists(output_file):
                return ("skipped", output_file, None)'''
            
            file_name_only = os.path.basename(output_file)
            if args.skip_existing and file_name_only in existing_files:
                return ("skipped", output_file, None)

            # Salvataggio
            torch.save(features, output_file)
            return ("saved", output_file, None)

        except Exception as e:
            # Non rilanciare: ritorna stato failed
            return ("failed", None, str(e))

    results = {"saved": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=args.save_workers) as executor:
        futures = [executor.submit(save_single, item) for item in features_list]

        for future in as_completed(futures):
            try:
                status, path, err = future.result()
                results[status] += 1

                if status == "failed":
                    print(f"Errore nel salvataggio: {err}")

            except Exception as e:
                # Caso raro: errore fuori da save_single
                results["failed"] += 1
                print(f"Errore imprevisto nel future: {e}")

    return results

'''

def save_features_batch(features_list, target_dir, dataset_name="codecfake"):
    """Salva features con naming corretto per dataset"""
    
    def save_single(item):
        features, filename, label, raw_idx, crop_idx, tag = item
        label = int(label)
        crop_idx = int(crop_idx)
        # Naming strategy diversa per dataset
        if dataset_name == "asvspoof2019":
            # Per ASVspoof: usa solo filename + crop (no indice numerico!)
            if crop_idx > 0:
                output_file = os.path.join(
                    target_dir,
                    f"{filename}_crop_{crop_idx}_label_{label}.pt"
                )
            else:
                output_file = os.path.join(
                    target_dir,
                    f"{filename}_label_{label}.pt"
                )
        else:
            # Per Codecfake: mantieni formato con indice se vuoi
            if crop_idx > 0:
                output_file = os.path.join(
                    target_dir,
                    f"{filename}_crop_{crop_idx}_label_{label}.pt"
                )
            else:
                output_file = os.path.join(
                    target_dir,
                    f"{filename}_label_{label}.pt"
                )
        
        # Skip se esiste già (se abilitato)
        if args.skip_existing and os.path.exists(output_file):
            return
        
        #torch.save(features.cpu().float(), output_file)
        torch.save(features.cpu().float(), output_file)
    
    # Salvataggio parallelo
    with ThreadPoolExecutor(max_workers=args.save_workers) as executor:
        executor.map(save_single, features_list)
        
'''
def process_dataset_optimized(dataset_name, dataset_path,output_base_dir, model_type,parts=["dev"]):
    """Processing ottimizzato con crops multipli per train"""
    
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()} dataset - OPTIMIZED")
    print('='*80)
    
    # Carica modello una sola volta
    print("\nLoading Wav2Vec2 model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        return_attention_mask=False
    )
    
    model, info = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        torch_dtype=torch.float16 if args.use_fp16 and cuda else torch.float32,
        use_safetensors=True, output_loading_info = True
        )
    
    print("Missing keys:", info["missing_keys"])
    print("Unexpected keys:", info["unexpected_keys"])
    print("Error msgs:", info["error_msgs"])
    
    if cuda:
        model = model.cuda()
    
    model.config.output_hidden_states = True
    model.eval()
    
    for part_ in parts:
        print(f"\n{'-'*60}")
        print(f"Processing {dataset_name} {part_}")
        print('-'*60)
        
        start_time = time.time()
        is_training = (part_ == "train")
        '''if part_ == 'eval':
            print("✓ Using 1 crop for evaluation")
            if dataset_name == "codecfake":
            raw_dataset = dataset.codecfake_eval(
                dataset_path,
                f"{dataset_path}/label",
                part=part_
            )
                num_crops = 3 if is_training else 1
            elif dataset_name == "asvspoof2019":
                raw_dataset = dataset.ASVspoof2019LAeval(
                    access_type="LA",
                    path_to_database=dataset_path,
                    path_to_protocol=f"{dataset_path}/ASVspoof2019_LA_cm_protocols",
                    part=part_
                )
                # NUOVO: ASVspoof ora usa la stessa strategia di Codecfake
                num_crops = 3 if is_training else 1
            elif dataset_name == "asvspoof5":
                if (is_training):
                    continue
                raw_dataset = dataset.ASVspoof5(
                    path_to_database=dataset_path,
                    path_to_protocol=f"{dataset_path}/protocols/",
                    part=part_
                )'''
        # IMPORTANTE: num_crops consistency
        # Train = 3 crops, Dev/Test = 1 crop
        if dataset_name == "codecfake":
            raw_dataset = dataset.codecfake(
                dataset_path,
                f"{dataset_path}/label",
                part=part_
            )
            num_crops = 3 if is_training else 1
        elif dataset_name == "asvspoof2019":
            raw_dataset = dataset.ASVspoof2019Raw(
                access_type="LA",
                path_to_database=dataset_path,
                path_to_protocol=f"{dataset_path}/ASVspoof2019_LA_cm_protocols",
                part=part_
            )
            # NUOVO: ASVspoof ora usa la stessa strategia di Codecfake
            num_crops = 3 if is_training else 1
        elif dataset_name == "asvspoof5":
            if (is_training):
                continue
            raw_dataset = dataset.ASVspoof5(
                path_to_database=dataset_path,
                path_to_protocol=f"{dataset_path}/protocols/",
                part=part_
            )
        
        if len(raw_dataset) == 0:
            print(f"No files found for {dataset_name} {part_}, skipping...")
            continue
        
        num_crops = 3 if is_training else 1
        print(f"Dataset size: {len(raw_dataset)} files")
        print(f"Crops per file: {num_crops}")
        print(f"Total samples to process: {len(raw_dataset) * num_crops}")
        print(f"MODEL TYPE : {model_type}")
        
        # Setup output directory
        target_dir = os.path.join(output_base_dir, dataset_name,model_type, part_, "xls")
        os.makedirs(target_dir, exist_ok=True)
        
        # Check existing files se skip_existing è abilitato
        existing_files = set()
        if args.skip_existing and os.path.exists(target_dir):
            existing_files = set(os.listdir(target_dir))
            if existing_files:
                print(f"Found {len(existing_files)} existing files in {target_dir}")
        
        # Reset dataset e dataloader per ogni parte (evita problemi di indici)
        audio_dataset = FastAudioDataset(
            raw_dataset, 
            processor, 
            num_crops=num_crops,
            dataset_name=dataset_name,
            part=part_
        )
        
        # DataLoader con configurazione ottimizzata
        dataloader = DataLoader(
            audio_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if args.num_workers > 0 else 0,
            collate_fn=collate_fn_fast,
            pin_memory=cuda and args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=True if args.num_workers > 0 else False,
            drop_last=False
        )
        
        total_processed = 0
        total_skipped = 0
        save_queue = []
        
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc=f"{dataset_name} {part_}") as pbar:
                for batch_idx, batch in enumerate(dataloader):
                    if batch is None:
                        pbar.update(1)
                        continue
                    
                    try:
                        # --- NUOVA LOGICA DI FILTRAGGIO PRE-INPUT ---
                        indices_to_process = []
                        for i in range(len(batch['filenames'])):
                            # Ricostruiamo il nome del file finale per il controllo
                            label_num = 1 if batch['labels'][i] in ["spoof", "fake", 1] else 0
                            crop_idx = int(batch['crop_indices'][i])
                            
                            if crop_idx > 0:
                                expected_name = f"{batch['filenames'][i]}_crop_{crop_idx}_label_{label_num}.pt"
                            else:
                                expected_name = f"{batch['filenames'][i]}_label_{label_num}.pt"
                            
                            if expected_name not in existing_files:
                                indices_to_process.append(i)
                        
                        # Se tutti i file del batch esistono già, saltiamo l'inferenza
                        if not indices_to_process:
                            total_skipped += len(batch['filenames'])
                            pbar.update(1)
                            continue

                        # Creiamo un "sub-batch" con solo i campioni mancanti
                        input_values = batch['input_values'][indices_to_process]
                        # input_values = batch['input_values']
                        
                        if cuda:
                            input_values = input_values.cuda(non_blocking=True)
                        
                        # Inference con mixed precision se abilitato
                        if args.use_fp16 and cuda:
                            with autocast():
                                outputs = model(input_values)
                                if model_type == "light":
                                    features = outputs.hidden_states[5]
                                else:
                                    features = outputs.last_hidden_state
                        else:
                            outputs = model(input_values)
                            if model_type == "light":
                                features = outputs.hidden_states[5]
                            else:
                                features = outputs.last_hidden_state
                        
                        # Prepara per salvataggio
                        feat_cpu = features.detach().cpu().float()
                    
                        '''batch_size = features.shape[0]
                        for i in range(batch_size):
                            
                            tag = batch['tags'][i] if batch['tags'][i] is not None else ""
                            save_queue.append((
                                feat_cpu[i],
                                #features[i],
                                batch['filenames'][i],
                                batch['labels'][i],
                                batch['raw_indices'][i],
                                batch['crop_indices'][i],
                                tag
                            ))'''
                        
                        # 3. Aggiungiamo alla save_queue usando gli indici corretti
                        for idx_in_subbatch, original_idx in enumerate(indices_to_process):
                            tag = batch['tags'][original_idx] if batch['tags'][original_idx] is not None else ""
                            save_queue.append((
                                feat_cpu[idx_in_subbatch].clone(),
                                batch['filenames'][original_idx],
                                batch['labels'][original_idx],
                                batch['raw_indices'][original_idx],
                                batch['crop_indices'][original_idx],
                                tag
                            ))
                        
                        # Salva in batch periodicamente
                        if len(save_queue) >= args.batch_size * 4:
                            save_stats = save_features_batch(save_queue, target_dir, dataset_name,existing_files)
                            total_processed += save_stats["saved"]
                            total_skipped += save_stats["skipped"]

                            if save_stats["failed"] > 0:
                                print(f"Attenzione: {save_stats['failed']} file non salvati")

                            save_queue = []
                        
                        '''if len(save_queue) >= args.batch_size * 4:
                            save_features_batch(save_queue, target_dir, dataset_name)
                            total_processed += len(save_queue)
                            save_queue = []'''
                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'processed': total_processed + len(save_queue),
                            'samples/s': f"{(total_processed + len(save_queue))/(time.time()-start_time):.1f}",
                            'batch_idx': batch_idx
                        })
                        
                        # Memory cleanup periodicamente
                        if batch_idx % 100 == 0 and cuda:
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"\nError in batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        pbar.update(1)
                        continue
                
                # Salva ultimi samples rimasti
                if len(save_queue) >= 0:
                    save_stats = save_features_batch(save_queue, target_dir, dataset_name,existing_files)
                    total_processed += save_stats["saved"]
                    total_skipped += save_stats["skipped"]

                    if save_stats["failed"] > 0:
                        print(f"Attenzione: {save_stats['failed']} file non salvati")

                    save_queue = []
                    
                '''if save_queue:
                    print(" \n --SAVE QUEUE -- \n")
                    save_features_batch(save_queue, target_dir, dataset_name)
                    total_processed += len(save_queue)'''
        # Cleanup dopo ogni parte
        del audio_dataset
        del dataloader
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✓ Completed {dataset_name} {part_}")
        print(f"  Processed: {total_processed} samples")
        print(f"  Skipped: {total_skipped} samples")
        print(f"  Time: {processing_time:.2f}s")
        print(f"  Throughput: {total_processed/processing_time:.2f} samples/s")
    
    # Final cleanup
    del model
    del processor
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

# Main execution
if __name__ == "__main__":
    print("\nStarting optimized feature extraction...")
    print("Settings: Train=3 crops, Dev=1 crop for both datasets")
    dataset_path = args.dataset_path
    output_path = args.output_path
    dataset_name = args.dataset
    model_type = args.model_type
    parts = ["eval"]  # Processa sia train che eval per entrambi i dataset
    if dataset_name == "codecfake":
        process_dataset_optimized(
            "codecfake",
            dataset_path,
            output_path,
            model_type,
            parts
        )
    
    if dataset_name == "asvspoof2019" :
        process_dataset_optimized(
            "asvspoof2019",
            dataset_path,
            output_path,
            model_type,
            parts
        )

    if args.dataset == "asvspoof5":
        process_dataset_optimized(
            "asvspoof5",
            dataset_path,
            output_path,
            model_type,
            parts
            
        )   
    
    print("\n" + "="*80)
    print("✓ Feature extraction completed!")
    print("✓ All files saved with correct naming")
    print("="*80)