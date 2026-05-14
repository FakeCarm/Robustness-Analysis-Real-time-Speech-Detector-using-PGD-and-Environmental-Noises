import raw_dataset as dataset
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)

if cuda:
    torch.cuda.set_per_process_memory_fraction(0.5)

def random_crop_or_pad(wav, target_length=16000, is_training=True):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    
    if waveform_len >= target_length:
        if is_training:
            max_start = waveform_len - target_length
            start_idx = np.random.randint(0, max_start + 1)
            return waveform[start_idx:start_idx + target_length]
        else:
            start_idx = (waveform_len - target_length) // 2
            return waveform[start_idx:start_idx + target_length]
    else:
        num_repeats = int(target_length / waveform_len) + 1
        padded_waveform = waveform.repeat(num_repeats)[:target_length]
        return padded_waveform

for part_ in ["train", "dev"]:
    print(f"\n{'='*60}")
    print(f"Processing {part_}")
    print('='*60)
    
    is_training = (part_ == "train")
    
    # Carica dataset
    codecspoof_raw = dataset.codecfake(
        "./Codecfake/",
        "./Codecfake/label/",
        part=part_
    )
    
    if len(codecspoof_raw) == 0:
        print(f"No files found for {part_}, skipping...")
        continue
    
    # Directory di output
    target_dir = os.path.join(
        "./features/codecfake_xls-r-5",
        part_,
        "xls-r-5"
    )
    
    # Carica modello
    print("Loading Wav2Vec2 model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
    model.config.output_hidden_states = True
    model.eval()
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Se è training, processa ogni file multiple volte con crop diversi
    num_crops_per_file = 3 if is_training else 1
    
    errors = 0
    with torch.no_grad():
        for idx in tqdm(range(len(codecspoof_raw)), desc=f"Extracting {part_} features"):
            try:
                waveform, filename, label = codecspoof_raw[idx]
                
                # Skip se waveform è dummy
                if waveform.sum() == 0:
                    errors += 1
                    continue
                
                # Genera multiple crops per training
                for crop_idx in range(num_crops_per_file):
                    # Random crop (training) o center crop (validation)
                    cropped_waveform = random_crop_or_pad(
                        waveform, 
                        target_length=16000,
                        is_training=is_training
                    )
                    
                    input_values = processor(
                        cropped_waveform, 
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_values.cuda()
                    
                    wav2vec2 = model(input_values).hidden_states[5].cpu()
                    
                    # Salva con indice del crop solo se multiple crops
                    if num_crops_per_file > 1:
                        output_file = os.path.join(
                            target_dir, 
                            f"{idx:06d}_{filename}_crop{crop_idx}_{label}.pt"
                        )
                    else:
                        output_file = os.path.join(
                            target_dir, 
                            f"{idx:06d}_{filename}_{label}.pt"
                        )
                    
                    torch.save(wav2vec2.float(), output_file)
                
                # Clear cache periodicamente
                if idx % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nError processing file {idx}: {e}")
                errors += 1
                continue
    
    actual_files = len(codecspoof_raw) - errors
    total_features = actual_files * num_crops_per_file
    print(f"Done with {part_}!")
    print(f"Processed {actual_files} audio files")
    print(f"Generated {total_features} feature files")
    print(f"Errors: {errors}")
    
    # Cleanup
    del model
    del processor
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("All preprocessing completed!")
print("="*60)