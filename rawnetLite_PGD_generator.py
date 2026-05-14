import argparse
from genericpath import exists
import numpy as np
import os
import torch
import torch.nn as nn
import sys
import torchaudio
from model import RawNetLite
from torch.utils.data import Dataset, DataLoader
import yaml
from yaml import Loader

from tqdm import tqdm
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from torch.utils.data import Dataset, DataLoader


from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art import config
from art.defences.preprocessor import Mp3Compression
from art.utils import get_file

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ----- Funzioni da utilizzare per caricamento in mono e normalizzazione tra -1 e 1, e per il padding/cropping deterministico a 1 secondo a 16kHz (16000 campioni) -----
def pad_dataset(wav, target_seconds=1.0, sample_rate=16000):

    waveform = wav.squeeze(0) if wav.dim() > 1 else wav
    waveform_len = waveform.shape[0]
    target_length = int(target_seconds * sample_rate)
    
    if waveform_len >= target_length:
        return waveform[:target_length]
    
    num_repeats = (target_length + waveform_len - 1) // waveform_len
    padded_waveform = waveform.repeat(num_repeats)[:target_length]
    return padded_waveform
def load_16k_mono(path, target_sr=16000):
    wav, sr = torchaudio.load(path)  # (C, T)
    # Portiamo in mono e a 16kHz se necessario
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = wav.float()     # (1,T)
    return wav, target_sr

def zero_mean_unit_var_norm_torch(x, attention_mask=None, padding_value=0.0, eps=1e-7):
    """
    x: (B, T) float
    attention_mask: (B, T) {0,1} oppure None
    """
    # Media 0 e varianza unitaria
    if attention_mask is None:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)  # ddof=0 come numpy.var()
        return (x - mean) / torch.sqrt(var + eps)

    mask = attention_mask.to(dtype=x.dtype)
    lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

    mean = (x * mask).sum(dim=-1, keepdim=True) / lengths
    var  = ((x - mean) ** 2 * mask).sum(dim=-1, keepdim=True) / lengths

    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_norm = x_norm * mask + padding_value * (1.0 - mask)
    return x_norm

def save_adversarial_samples(adv_np, filenames, adversarial_sample_path, iter, ep):
    ep_adv = os.path.join(adversarial_sample_path, f"ep_{ep}")
    os.makedirs(ep_adv, exist_ok=True)
    full_path = os.path.join(ep_adv,f"iter_{iter}")
    os.makedirs(full_path, exist_ok=True)
    for adv_sample, filename in zip(adv_np, filenames):
        output_path = os.path.join(full_path, f"{filename}_{ep}_{iter}.flac")
        torchaudio.save(output_path, torch.from_numpy(adv_sample).unsqueeze(0), 16000)

#SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
class ASVspoof5(Dataset):
    def __init__(self, path_to_database, path_to_protocol, part='train'):

        super(ASVspoof5, self).__init__()
        self._path_to_database = path_to_database
        self._path_to_protocol = path_to_protocol
        self._path_to_audio = os.path.join(self._path_to_database, part,"flac_E_eval")
        self._part = part
        protocol = os.path.join(self._path_to_protocol, 'ASVspoof5' + '.' + self._part + '.tsv')
        print(f"Loading ASVspoof5 {part} dataset from {protocol}")
        self._formato = ".flac"

        #print("CWD =", os.getcwd())
        #print(os.listdir(self._path_to_protocol)[:5])
        #os.path.exists(self._path_to_protocol) or print(f"WARNING: Protocol file not found: {self._path_to_protocol}"); return KeyError("Protocol file not found")
        
        self.all_info = []
        total = 0
        total_spoof = 0

        with open(protocol, 'r') as f:
            lines = f.readlines()
            
        
        for line in lines:
            #if total_spoof == 1:
            #    break
            total += 1
            parts = line.strip().split(' ')
            if len(parts) == 10:
                filename = parts[1]
                label = parts[8]
                id_attack = parts[7]
                filepath = os.path.join(self._path_to_audio, filename + self._formato)
                if label == "spoof":
                    total_spoof += 1   
                    if os.path.exists(filepath):
                        self.all_info.append((filename, label, id_attack))


        print(f"Dataset ASVspoof5 {part}: Found {len(self.all_info)}/{total_spoof} spoof files on a total of {total} files")
    
    def __len__(self):
        return len(self.all_info)
    
    def __getitem__(self, idx):
        filename, label, id_attack = self.all_info[idx]
        filepath = os.path.join(self._path_to_audio, filename + self._formato)
        #print(f"Loading file: {filepath} (ID Attack: {id_attack}, Label: {label})")
        y = 0 if label == "bonafide" else 1
        try:
            waveform, sr = load_16k_mono(filepath, target_sr=16000)
            waveform = pad_dataset(waveform, target_seconds=1.0, sample_rate=sr)
            return waveform, filename, y, id_attack

        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, 16000), filename, y, id_attack
        
    
def reshape_features_for_model(features, model_name):
    """Adatta le features al modello specifico"""
    # features arriva come [batch, seq_len, hidden_dim] = [64, 49, 1024]
    if model_name == 'W2VAASIST':
        # W2VAASIST vuole [batch, hidden_dim, seq_len] NON 4D!
        return features.transpose(1, 2)  # [64, 1024, 49]
    else:
        return features

class spoofing_pipeline(nn.Module):
    def __init__(self,wav2vec2, rawnetlite, model_name):

        super().__init__()
    
        self.frontend = wav2vec2
        self.backend = rawnetlite
        self.model_name = model_name

        if self.model_name.lower() != "aasist":
            self.frontend.feature_extractor._freeze_parameters()

        for p in self.frontend.parameters():
            p.requires_grad = False

        # congela rawnet
        for p in self.backend.parameters():
            p.requires_grad = False


    def forward(self, waveform):
        # Ottengo waveform paddata e in formato mono 16kHz
        #waveform, sr = load_16k_mono(waveform)

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        waveform_norm = zero_mean_unit_var_norm_torch(waveform)

        #mid_features = self.wav2vec2(waveform_norm).hidden_states[5]
        mid_features = self.frontend(waveform_norm).last_hidden_state
        if self.model_name.lower() == "aasist":
            mid_features = reshape_features_for_model(mid_features, 'W2VAASIST')
        _, logits = self.backend(mid_features)
        return logits

def snr_db(x_adv: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
    """
    x, x_adv: shape (B, T) o (B, 1, T)
    ritorna: snr per campione (B,)
    """
    if x.dim() == 3:
        x = x.squeeze(1)
        x_adv = x_adv.squeeze(1)

    delta = x_adv - x
    p_signal = torch.sum(x**2, dim=1) 
    p_noise  = torch.sum(delta**2, dim=1) + eps
    return 10.0 * torch.log10(p_signal / p_noise)

def generate_audio(weights_model_path, use_fp16, dataset_path, output_dir="./adversarial_results/", adversarial_sample_path="./adversarial_samples/"):
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(adversarial_sample_path, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #dataset = ASVspoof5(dataset_path, dataset_path +'/protocols', "eval")
    
    # Carico i modelli    
    try:
        
        MODEL = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",revision="refs/pr/15", use_safetensors=True,
        )
        MODEL.config.output_hidden_states = True
        
        k = 5
        # tengo solo i primi k layer transformer
        MODEL.encoder.layers = nn.ModuleList(list(MODEL.encoder.layers)[:k])
        # aggiorno anche la config per coerenza
        MODEL.config.num_hidden_layers = k
        if MODEL.config.do_stable_layer_norm:
            MODEL.encoder.layer_norm = nn.Identity()
            
        MODEL.eval() 




        ADD_MODEL = RawNetLite()
        ADD_MODEL = torch.load(weights_model_path, weights_only = False)
        


        PIPELINE = spoofing_pipeline(MODEL, ADD_MODEL, "rawnet").to(device)
        PIPELINE.eval()



        # AASIST PIPELINE
        wav2vec2 = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",revision="refs/pr/15", use_safetensors=True,
        )
        wav2vec2.config.output_hidden_states = True
        
        k = 5
        # tengo solo i primi k layer transformer
        wav2vec2.encoder.layers = nn.ModuleList(list(wav2vec2.encoder.layers)[:k])
        # aggiorno anche la config per coerenza
        wav2vec2.config.num_hidden_layers = k
        if wav2vec2.config.do_stable_layer_norm:
            wav2vec2.encoder.layer_norm = nn.Identity()
            
        wav2vec2.eval()

        model_path = "../Models/cotrain_W2VAASIST_csam_v3/anti-spoofing_feat_model.pt"
        aasist = torch.load(model_path, weights_only=False)
        
        PIPELINE_TARGET = spoofing_pipeline(wav2vec2, aasist, "aasist").to(device)
        PIPELINE_TARGET.eval()
        

    except Exception as e:
        print(f"   ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        return
    

    dataset = ASVspoof5(dataset_path, dataset_path +'/protocols', "eval")
    cropped_dataset = DataLoader(dataset,batch_size=32, shuffle=False, num_workers=4)
    dataset_len = len(dataset)


    classifier_art = PyTorchClassifier(
        model=PIPELINE,
        loss=nn.CrossEntropyLoss(),
        input_shape=(16000,),
        nb_classes=2,
        device_type="gpu" if cuda else "cpu"
    )

    # Definiamo il range massimo di pertubazione
    eps_range = 0.4
    # Definiamo l'aggiunta di pertubazione ad ogni step
    #eps_steps = [0.1, 0.01, 0.0001]
    # Definiamo il numero di step
    #max_iters = [1,3,5]
    # !!!! IN REALTA RAPPRESENTA EPS !!! EPS STEP E' DATO DA EPS/ITERS
    eps = [1e-8,1e-5, 0.0001, 0.001, 0.005, 0.01, 0.1]
    #eps = [1e-8,1e-5, 0.0001]
    #eps_steps = [0.05, 0.01, 0.001, 0.0001]
    # Definiamo il numero di step
    max_iters = [1,3,7]
    #max_iters = [1,3]

    # inizializza una lista per ogni posizione della lista score_by_iter_step[][]
    scores_by_iter_step_adv = [[[] for _ in range(len(eps))]
                       for _ in range(len(max_iters))]
    
    # Inizializza matrice per perturbazioni medie per ogni combinazione di max_iter ed eps_step
    perturbations = np.zeros((len(max_iters), len(eps)))

    snr = np.zeros((len(max_iters), len(eps)))

    p10 = np.zeros((len(max_iters), len(eps)))
    p90 = np.zeros((len(max_iters), len(eps)))


    flag_filename_list = True
    filenames_list = []
    all_labels = []
    genuine_threshold = 0.70

    for n_iter, max_iter in enumerate(max_iters):
        # Inizializzo la lista di scores
        alpha = 9999
        for n_eps, ep in enumerate(eps):
            alpha = ep/max_iter
            print(f"Generating adversarial samples with max_iter={max_iter} and eps={ep} and eps_step={alpha}") 
           
            sum_perturbation = 0
            count = 0
            snr_for_batch_samples = []


            for batch in tqdm(cropped_dataset):
                if batch is None:
                        continue
                
                # Ottengo il batch
                cropped, filenames, labels, id_attack = batch
                
                # Porto i campioni su GPU
                cropped_dev = cropped.to(device)
                #Inizializzo l'attacco sul modello
                attack = ProjectedGradientDescentPyTorch(estimator=classifier_art,
                                                            eps = ep, eps_step=alpha,
                                                                targeted=False, max_iter = max_iter, verbose = False)
                
                #Porto il campione su cpu in formpato numpy staccandolo dal gradiente.
                cropped_for_art = cropped.detach().numpy().astype(np.float32)

                # genera adversarial (ritorna numpy)
                adv_np = attack.generate(cropped_for_art)

                # SALVATAGGIO ADVERSARIAL
                
                #save_adversarial_samples(adv_np, filenames, adversarial_sample_path, max_iter, ep)
                
                # torna a torch e rimetti su GPU
                adv = torch.from_numpy(adv_np).to(device=device, dtype=cropped_dev.dtype)
                snr_values_dev = snr_db(adv, cropped_dev)

                # Valuto su campione normale
                #outputs = PIPELINE_TARGET(cropped_dev)
                #print("output su NON avversarial ottenuto")
                # Valuto adversarial
                outputs_adversarial = PIPELINE_TARGET(adv)
                
                #print("Original dim", outputs.shape)

                
                # Se il modello restituisce logit per 2 classi, applichiamo softmax e prendiamo la probabilità della classe "real" (index 0) e "fake" (index 1)
                #probs = torch.softmax(outputs, dim=1)
                #print("Probs dim", probs.shape)
                #batch_scores_by_fake = probs[:, 1].cpu().numpy() # Normalizza i logit e prendi la probabilità della classe "fake" e la salva in scores per ogni campione      
                
                # Faccio inferenza sull'adversarial
                probs_adv = torch.softmax(outputs_adversarial, dim=1)
                batch_scores_by_fake_adv = probs_adv[:, 1].cpu().numpy() # Normalizza i logit e prendi la probabilità della classe "fake" e la salva in scores per ogni campione

                # Salvo i nomi dei file in una lista seguendo gli altri file
                if flag_filename_list == True:
                    filenames_list.extend(filenames)
                    all_labels.extend(labels.numpy())
                # Calcolo la perturbazione per ogni campione
                delta = np.abs(cropped - adv_np)
                # Faccio la media delle pertubazioni per ogni campione
                mean_per_sample = delta.mean(axis=1)  # ((B,) ottengo la media per ogni campione
                # Sommo la pertubazione per ogni campione
                sum_perturbation += mean_per_sample.sum().item()
                # Calcolo il numero di valori di pertubazione (dovrebbe essere uguale al numero di campioni)
                count += mean_per_sample.numel()

                
                snr_for_batch_samples.append(snr_values_dev.detach())  # (B,) su GPU
                #snr_for_batch_samples.extend(snr_values_dev.cpu().numpy().tolist())

                # Salvo gli score ottenuti
                scores_by_iter_step_adv[n_iter][n_eps].extend(batch_scores_by_fake_adv.tolist())

                #scores_by_iter_step[n_iter][n_eps].extend(batch_scores_by_fake.tolist())
              
              
            print(f"\n \n ------  Fine Generazione ep{ep} iter{iter} ---------- \n \n ")

            
            perturbations[n_iter][n_eps]= sum_perturbation / count

            snr_all = torch.cat(snr_for_batch_samples, dim=0)      # su GPU
            median_snr = torch.nanmedian(snr_all).item()           # .item() porta solo uno scalare in CPU
            p10[n_iter][n_eps] = torch.quantile(snr_all[~torch.isnan(snr_all)], 0.10).item()
            p90[n_iter][n_eps] = torch.quantile(snr_all[~torch.isnan(snr_all)], 0.90).item()            
            snr[n_iter][n_eps] = median_snr
            flag_filename_list = False

            # Inizio fase di valutazione per ogni combinazione di max_iter ed eps_step
            all_labels = np.array(all_labels)
    

            # Numero di campioni classificati correttamente
            correctly_classified = 0
            # Numero di campioni classificati erroneamente
            correctly_misclassified = 0
            # Numero campioni misclassificati come bonafide  ma che in realtà sono spoof 
            misclassificazione_inversa = 0

        
            num_samples = len(filenames_list)
            if (num_samples == 0):
                print("Errore: Nessun campione processato. Verifica il caricamento del dataset e la generazione degli adversarial.")
                exit(1)
            eer = None
            eer_min_threshold = None
            roc_auc = None

            # Inizializzo le metriche per questa combinazione di max_iter ed eps_step
            # Correttamente classificati x accuracy
            correctly_classified = 0
            # Correttamente misclassificati (originale predice fake ma adversarial bonafide)
            correctly_misclassified = 0
            # Errori di misclassificazione inversa (il campione è bonafide ma viene classificato come spoof)
            misclassificazione_inversa = 0
            # Entrambi classificano bonafide un audio spoof
            same_between_adv_and_original = 0

            accuracy = 0.0
            attack_success_rate = 0.0
            misclassificazione_inversa_rate = 0.0
            same_between_adv_and_original_rate = 0.0
            

            # Scores sugli adversarial
            scores_adv = np.array(scores_by_iter_step_adv[n_iter][n_eps])
            # Scores su campioni originali

            if (len(scores_adv) != len(all_labels)):
                exception_message = f"Errore: Scores e labels hanno lunghezze diverse. Scores: {len(scores_adv)}, Labels: {len(all_labels)}"
                print(exception_message)
                exit(1)

            #real_scores_adv = scores_adv[all_labels == 0]
            #fake_scores_adv = scores_adv[all_labels == 1]
            # Utilizzo come target la classe 1 ovvero la classe spoof.
            #eer, eer_thresholds, frr, far, min_index  = em.compute_eer(fake_scores,real_scores)
            #eer_min_threshold = eer_thresholds[min_index]

            # Predizioni del modello sugli adversarial
            predictions_adv = (scores_adv >= genuine_threshold).astype(int)
            # Predizioni del modello sui campioni originali
            
            '''
            filename_misclassified = []

            for pred_adv, pred, label, filenames in zip(predictions_adv, prediction, all_labels, filenames_list):
                if filenames is None:
                    print("Warning: Filename is None for a sample. Skipping this sample.")
                    continue
                if pred_adv == label: # Se il modello indovina
                    correctly_classified += 1
                else: # Se il modello misclassifica l'adversarial
                    if label == 1: # Se il campione originale è bonafide
                        # pred_adv 0 e pred 1
                        if pred_adv != pred: # Se il modello classifica correttamente il campione originale ma non l'adversarial, cio significa che l'adversarial ha avuto successo nel far cambiare la predizione del modello, quindi è una misclassificazione "corretta" da parte dell'attacco.
                            correctly_misclassified += 1    # Misclassificazione corretta solo se misclassifica come bonafide
                            filename_misclassified.append(filenames)
                        else:
                            # pred_adv 0 e pred 0
                            same_between_adv_and_original += 1
                    else:
                        # Qui non dovrebbe entrare mai se si classificano solo spoof
                        misclassificazione_inversa +=1
                        
            
            if correctly_classified > 0:
                accuracy = correctly_classified / (num_samples) * 100
            else:
                accuracy = 0.0
            
            if correctly_misclassified > 0:
                attack_success_rate = correctly_misclassified / (num_samples) * 100
            else :
                attack_success_rate = 0.0
            
            if misclassificazione_inversa > 0:
                misclassificazione_inversa_rate = misclassificazione_inversa / (num_samples) * 100
            else:
                misclassificazione_inversa_rate = 0.0

            if same_between_adv_and_original > 0:
                same_between_adv_and_original_rate = same_between_adv_and_original / (num_samples) * 100
            else:
                same_between_adv_and_original_rate = 0.0
            #fpr, tpr, roc_thresholds = roc_curve(all_labels, scores_adv, pos_label=1)  
            #roc_auc = auc(fpr, tpr)

            # 6. SALVA RISULTATI
            results = {
                'Iterazione': max_iter,
                'Eps ': ep,
                'Eps_step' : alpha,
                'EER (%)': eer * 100 if eer is not None else 9999,
                'AUC': roc_auc if roc_auc is not None else 9999,
                'EER Threshold': eer_min_threshold is not None and eer_min_threshold or 9999,
                'Threshold Modello Originale': genuine_threshold,
                'Accuracy (%)': accuracy,
                '# Correctly Classified ': int(correctly_classified),
                'Accuracy Misclassified (%)': attack_success_rate,
                '# Misclassificazione Corretta ': int(correctly_misclassified),
                'Misclassificazione Inversa (%)': misclassificazione_inversa_rate,
                '# Misclassificazione Inversa ': int(misclassificazione_inversa),
                'Same Between Adv and Original (%)': same_between_adv_and_original_rate,
                '# Same Between Adv and Original ': int(same_between_adv_and_original),
                'Perturbazione Media': perturbations[n_iter][n_eps],
                'SNR Mediana (dB)': snr[n_iter][n_eps],
                'SNR 10° Percentile (dB)': p10[n_iter][n_eps],
                'SNR 90° Percentile (dB)': p90[n_iter][n_eps],
                'Total Samples': len(all_labels)
            }

            # Salva risultati in file
            with open(os.path.join(output_dir, f'metrics_{max_iter}_{ep}.txt'), 'w') as f:
                f.write("="*10 + "Model evaluation results" + "-" *10 + "\n")
                
                
                f.write("-"*10 + "PERFORMANCE METRICS:" + "-"*10 + "\n")
                for key, value in results.items():
                    if 'Time' not in key and 'Throughput' not in key and 'Total' not in key:
                        f.write(f"{key:<30}: {value}\n")
                    else:
                        f.write(f"{key:<30}: {value}\n")
                
                f.write("\n")
                f.write("-"*10 + "Misclassified Samples:" + "-"*10 + "\n")
                for filename in filename_misclassified:
                    f.write(f"{filename}\n")
            
            #print(f" lunghezza filenames_list: {len(filenames_list)}, lunghezza all_labels: {len(all_labels)}, lunghezza scores_by_iter_step_adv: {len(scores_by_iter_step_adv[i][j])}")
            '''

            results = {
                'Iterazione': max_iter,
                'Eps ': ep,
                'Eps_step' : alpha,
                'Threshold Modello Originale': genuine_threshold,
                'Perturbazione Media': perturbations[n_iter][n_eps],
                'SNR Mediana (dB)': snr[n_iter][n_eps],
                'SNR 10° Percentile (dB)': p10[n_iter][n_eps],
                'SNR 90° Percentile (dB)': p90[n_iter][n_eps],
                'Total Samples': len(all_labels)
            }

            # Salva risultati in file
            with open(os.path.join(output_dir, f'metrics_{max_iter}_{ep}.txt'), 'w') as f:
                f.write("="*10 + "Model evaluation results" + "-" *10 + "\n")
                
                
                f.write("-"*10 + "PERFORMANCE METRICS:" + "-"*10 + "\n")
                for key, value in results.items():
                    if 'Time' not in key and 'Throughput' not in key and 'Total' not in key:
                        f.write(f"{key:<30}: {value}\n")
                    else:
                        f.write(f"{key:<30}: {value}\n")
                
            with open(os.path.join(output_dir, f'sample_probability_adversarial_Iter{max_iters[n_iter]}_Eps{eps[n_eps]}.txt'), 'w') as f:
                for filename, prediction_adv, score in zip(filenames_list, predictions_adv, scores_by_iter_step_adv[n_iter][n_eps]):
                    f.write(f"{filename} {score:.6f} {prediction_adv}\n")

if __name__ == "__main__":

    weights_model_path = "../Models/cotrain_rawnet_csam_v2/anti-spoofing_feat_model.pt"
    dataset_path = "../ASVspoof5"
    adversarial_sample_path = "./adversarial_samples_RAWNET/"
    output_dir = "./adversarial_results_RAWNET"
    os.makedirs(adversarial_sample_path, exist_ok = True)
    use_fp16 = True
    generate_audio(weights_model_path, use_fp16, dataset_path, output_dir=output_dir, adversarial_sample_path=adversarial_sample_path)
