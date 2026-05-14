import glob
from html import parser
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import librosa
import argparse
import torchaudio
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import eval_metrics as em
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score

from transformers import Wav2Vec2Model
import math
# set seed
np.random.seed(123)


'''class AddFade(nn.Module):
    def __init__(self, max_fade_size=.5, fade_shape=None, fix_fade_size=False):
        super(AddFade, self).__init__()
        self.max_fade_size = max_fade_size
        self.fade_shape = fade_shape
        self.fix_fade_size = fix_fade_size
    def add_fade(self, audio, fade_in_len, fade_out_len, fade_shape):
        fade_transform = torchaudio.transforms.Fade(fade_in_len=fade_in_len, fade_out_len=fade_out_len, fade_shape=fade_shape)
        return fade_transform(audio)
    def forward(self, audio, fade_in_len=None, fade_out_len=None, fade_shape=None):
        wave_length = audio.size()[-1]
        # wave_length = audio.shape[1] 
        if fade_in_len == None:
            if self.fix_fade_size:
                fade_in_len = int(self.max_fade_size  * wave_length)
            else:
                fade_in_len = random.randint(0, int(self.max_fade_size  * wave_length))
        if fade_out_len == None:
            if self.fix_fade_size:
                fade_out_len = int(self.max_fade_size  * wave_length)
            else:
                fade_out_len = random.randint(0, int(self.max_fade_size  * wave_length))
        if fade_shape == None:
            if self.fade_shape == None:
                fade_shape = random.choice(["quarter_sine", "half_sine", "linear", "logarithmic", "exponential"])
            else:
                fade_shape = self.fade_shape
        return self.add_fade(audio, fade_in_len, fade_out_len, fade_shape)
'''

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

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

# ----- Funzioni da utilizzare per caricamento in mono e normalizzazione come fa aasist, e per il padding/cropping deterministico a 1 secondo a 16kHz (16000 campioni) -----
def pad_dataset(wav, target_seconds=1.0, sample_rate=16000, crop = "center_crop"):

    waveform = wav.squeeze(0) if wav.dim() > 1 else wav
    waveform_len = waveform.shape[0]
    target_length = int(target_seconds * sample_rate)

    if waveform_len >= target_length:
        if crop == "center_crop":
            start_idx = (waveform_len - target_length) // 2
            return waveform[start_idx : start_idx + target_length]
        
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
            '''if total_spoof == 10:
                break'''
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
            #output_path = os.path.join("../ASVspoof5/eval/flac_E_eval_1sec", f"{filename}_1sec.flac")
            #torchaudio.save(output_path, waveform.unsqueeze(0), 16000)
            return waveform, filename, y, id_attack
    
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, 16000), filename, y, id_attack

# Estrae ultimo layer di wav2vdec2 e lo passa ad AASIST, senza aggiornare i pesi di nessuno dei due modelli
class spoofing_pipeline(nn.Module):
    def __init__(self,wav2vec2, aasist, layer_index = 4):

        super().__init__()
        self.layer_index = layer_index
        self.wav2vec2 = wav2vec2
        self.aasist = aasist

        for p in self.wav2vec2.parameters():
            p.requires_grad = False

        # congela AASIST
        for p in self.aasist.parameters():
            p.requires_grad = False
    
    def forward(self, waveform):
        # Ottengo waveform paddata e in formato mono 16kHz
        #waveform, sr = load_16k_mono(waveform)

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        waveform_norm = zero_mean_unit_var_norm_torch(waveform)

        #mid_features = self.wav2vec2(waveform_norm).hidden_states[-1]
        mid_features = self.wav2vec2(waveform_norm).last_hidden_state
        mid_features = reshape_features_for_model(mid_features, 'W2VAASIST')
        _, logits = self.aasist(mid_features)
        return logits

def reshape_features_for_model(features, model_name):
    """Adatta le features al modello specifico"""
    # features arriva come [batch, seq_len, hidden_dim] = [64, 49, 1024]
    if model_name == 'W2VAASIST':
        # W2VAASIST vuole [batch, hidden_dim, seq_len] NON 4D!
        return features.transpose(1, 2)  # [64, 1024, 49]
    else:
        return features

def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds

    p_target = 1- Pspoof
    for i in range(0, len(frr)):
        # Weighted sum of false negative and false positive errors.
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def save_adversarial_samples(adv_np, filenames, adversarial_sample_path, n_iter, n_step, sdb):
    for adv_sample, filename in zip(adv_np, filenames):
        output_path = os.path.join(adversarial_sample_path, f"adv_iter{n_iter}_step{n_step}_{filename}_{sdb}.flac")
        torchaudio.save(output_path, torch.from_numpy(adv_sample).unsqueeze(0), 16000)

def snr_db(x_adv: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
    """
    x, x_adv: shape (B, T) oppure (B, 1, T)
    ritorna: SNR per campione, shape (B,)
    """
    if x_adv.shape != x.shape:
        raise ValueError(f"Shapes of x_adv {x_adv.shape} and x {x.shape} must match")

    delta = x_adv - x

    x_flat = x.reshape(x.shape[0], -1)
    delta_flat = delta.reshape(delta.shape[0], -1)

    p_signal = torch.sum(x_flat ** 2, dim=1)
    p_noise = torch.sum(delta_flat ** 2, dim=1) + eps

    return 10.0 * torch.log10(p_signal / p_noise)

def fusion (orig, noise, sr_og, alpha = 0.5):
    # orig, noise: (T,) numpy array
    # alpha: peso per il rumore (0.5 = mix bilanciato)

    
    orig = pad_dataset(orig, target_seconds=1.0, sample_rate=sr_og)
    noise = pad_dataset(noise, target_seconds=1.0, sample_rate=sr_og)
    fused = (1 - alpha) * orig + alpha * noise
    return fused, orig

def fit_noise_audio_to_x(
    x: torch.Tensor,
    noise_audio: torch.Tensor,
    crop_mode: str = "repeat"
):
    """
    Adatta un audio-rumore alla shape di x.

    Parametri
    ---------
    x : torch.Tensor
        shape (B, T) oppure (B, 1, T)

    noise_audio : torch.Tensor
        può avere shape:
        - (Tn,)
        - (1, Tn)
        - (B, Tn)
        - (B, 1, Tn)

    crop_mode : str
        - "repeat": se il rumore è più corto, lo ripete fino a coprire T
        - "center_crop": se il rumore è più lungo, prende il centro
        - "random_crop": se il rumore è più lungo, prende una finestra casuale

    Ritorna
    -------
    noise_fitted : torch.Tensor
        stessa shape di x
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"x must have shape (B,T) or (B,1,T), got {x.shape}")

    B = x.shape[0]
    T = x.shape[-1]

    # Porta noise_audio a shape (Bn, Tn)
    if noise_audio.ndim == 1:
        noise_flat = noise_audio.unsqueeze(0)         # (1, Tn)
    elif noise_audio.ndim == 2:
        noise_flat = noise_audio                      # (Bn, Tn)
    elif noise_audio.ndim == 3:
        noise_flat = noise_audio.reshape(noise_audio.shape[0], -1)
    else:
        raise ValueError(f"Unsupported noise_audio shape: {noise_audio.shape}")

    Bn, Tn = noise_flat.shape

    # Se c'è un solo noise audio, lo condivido per tutto il batch
    if Bn == 1:
        noise_flat = noise_flat.repeat(B, 1)
    elif Bn != B:
        raise ValueError(
            f"noise_audio batch size must be 1 or equal to x batch size. "
            f"Got noise batch={Bn}, x batch={B}"
        )

    fitted = []
    for i in range(B):
        n = noise_flat[i]

        if Tn == T:
            n_fit = n

        elif Tn < T:
            # ripeti fino a raggiungere T
            reps = math.ceil(T / Tn)
            n_fit = n.repeat(reps)[:T]

        else:  # Tn > T
            if crop_mode == "center_crop":
                start = (Tn - T) // 2
            elif crop_mode == "random_crop":
                start = torch.randint(0, Tn - T + 1, (1,), device=n.device).item()
            else:
                # default: prendi l'inizio
                start = 0
            n_fit = n[start:start + T]

        fitted.append(n_fit)

    noise_fitted = torch.stack(fitted, dim=0)  # (B, T)

    if x.ndim == 3:
        noise_fitted = noise_fitted.unsqueeze(1)  # (B, 1, T)

    return noise_fitted

def make_audio_with_target_snr_from_noise_audio(
    x: torch.Tensor,
    noise_audio: torch.Tensor,
    snr_target_db,
    crop_mode: str = "repeat",
    eps: float = 1e-12
):
    """
    Costruisce x_adv = x + delta usando come perturbazione un audio reale (es. 1 secondo),
    scalato in modo da ottenere un SNR target scelto a priori.

    Parametri
    ---------
    x : torch.Tensor
        shape (B, T) oppure (B, 1, T)

    noise_audio : torch.Tensor
        audio-rumore base, es. 1 secondo
        shape supportate:
        - (Tn,)
        - (1, Tn)
        - (B, Tn)
        - (B, 1, Tn)

    snr_target_db : float | list | torch.Tensor
        SNR desiderato in dB
        - float: stesso SNR per tutto il batch
        - list/tensor di lunghezza B: SNR diverso per campione

    crop_mode : str
        strategia per adattare il noise audio alla lunghezza di x

    Ritorna
    -------
    x_adv : torch.Tensor
    delta : torch.Tensor
    actual_snr : torch.Tensor
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"x must have shape (B,T) or (B,1,T), got {x.shape}")

    B = x.shape[0]

    # 1) adatta il noise audio alla lunghezza di x
    noise = fit_noise_audio_to_x(x, noise_audio, crop_mode=crop_mode)

    # 2) flatten per calcolo potenze
    x_flat = x.reshape(B, -1) # se (b,1,16k) trasforma in (B,16k)
    noise_flat = noise.reshape(B, -1)

    p_signal = torch.sum(x_flat ** 2, dim=1, keepdim=True)               # (B,1)
    p_noise_raw = torch.sum(noise_flat ** 2, dim=1, keepdim=True) + eps  # (B,1)

    # 3) gestisci SNR target
    if isinstance(snr_target_db, (float, int)):
        snr_target_db = torch.full(
            (B, 1), float(snr_target_db), device=x.device, dtype=x.dtype
        )
    else:
        snr_target_db = torch.as_tensor(
            snr_target_db, device=x.device, dtype=x.dtype
        ).reshape(B, 1)

    # 4) conversione dB -> lineare
    target_linear = 10.0 ** (snr_target_db / 10.0)

    # 5) scala il noise audio
    alpha = torch.sqrt(p_signal / (p_noise_raw * target_linear))         # (B,1)

    delta_flat = alpha * noise_flat
    delta = delta_flat.reshape_as(x)

    # 6) audio finale
    x_adv = x + delta

    # 7) verifica
    actual_snr = snr_db(x_adv, x, eps=eps)

    return x_adv, delta, actual_snr


def generate_audio(model_path, use_fp16, dataset_path, output_dir, adversarial_sample_path, snr_target_db, nome_rumore, model_type):
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(adversarial_sample_path, exist_ok=True)
    
    file_path = os.path.join(output_dir, f'sample_probability_ambient_adversarial_SNR{snr_target_db}.txt')
    # 2. Controlla se NON esiste
    if os.path.exists(file_path):
        exit("File con SNR già esistente")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #dataset = ASVspoof5(dataset_path, dataset_path +'/protocols', "eval")
    
    # Carico i modelli    
    try:
    
        MODEL = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",revision="refs/pr/15", use_safetensors=True,
        )
        MODEL.config.output_hidden_states = True

        # se scelgo la versione light, prendo solo i primi k layer di wav2vec2 e disabilito lo stable layer norm, che è inutile se ho pochi layer e rallenta molto l'inferenza  
        if model_type == "light":
            print("MODEL LIGHT")
            k = 5
            # tengo solo i primi k layer transformer
            MODEL.encoder.layers = nn.ModuleList(list(MODEL.encoder.layers)[:k])
            # aggiorno anche la config per coerenza
            MODEL.config.num_hidden_layers = k
            if MODEL.config.do_stable_layer_norm:
                MODEL.encoder.layer_norm = nn.Identity()
        print("MODEL FULL")
        MODEL.to(device)
        MODEL.eval()    
    
        # Carico l'intero modello compreso di pesi .pt
        ADD_MODEL = torch.load(model_path, map_location=device, weights_only=False)
        ADD_MODEL.eval()

        PIPELINE = spoofing_pipeline(MODEL, ADD_MODEL).to(device)
        PIPELINE.eval()
        

    except Exception as e:
        print(f"   ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    dataset = ASVspoof5(dataset_path, dataset_path +'/protocols', "eval")
    cropped_dataset = DataLoader(dataset,batch_size=32, shuffle=False, num_workers=4)

    
    path_noisy_audio = f"../ESC-50/train/{nome_rumore}.wav"
    noisy_audio, sr = load_16k_mono(path_noisy_audio, 16000)
    noisy_audio = pad_dataset(noisy_audio, sample_rate=sr) 
    ''' os.makedirs(adversarial_sample_path + f"/{nome_rumore}", exist_ok=True)
    output_path = os.path.join(adversarial_sample_path + f"/{nome_rumore}", f"{nome_rumore}_1sec.flac")
    torchaudio.save(output_path, noisy_audio.unsqueeze(0), 16000)
    '''
    noisy_audio_dev = noisy_audio.to(device)

    scores = []
    if model_type == "full":
        genuine_threshold = 0.714
    else:
        genuine_threshold = 0.7
    filenames_list = []
    all_labels = []
    #fused = zero_mean_unit_var_norm_torch(fused)
    # evaluate fused audio
    # compute differences between audio
    with torch.no_grad():
        for batch in tqdm(cropped_dataset):
            if batch is None:
                continue
            # Ottengo il batch
            og, filenames, labels, id_attack = batch

            filenames_list.extend(filenames)
            all_labels.extend(labels)
            og_dev = og.to(device)
            
            
            # In realtà viene fatto un center crop in quanto gia viene croppato prima con la funzione pad dataset
            x_adv, delta, actual_snr = make_audio_with_target_snr_from_noise_audio(x=og_dev, noise_audio=noisy_audio_dev, snr_target_db=snr_target_db, crop_mode="repeat")
            #save_adversarial_samples(x_adv.detach().cpu().numpy(),filenames_list,adversarial_sample_path,0,0, snr_target_db)
            outputs = PIPELINE(x_adv)
            #outputs = PIPELINE(og_dev)
            probs = torch.softmax(outputs, dim=1)
            batch_scores_by_fake = probs[:, 1].cpu().numpy() # Normalizza i logit e prendi la probabilità della classe "fake" e la salva in scores per ogni campione      
            scores.extend(batch_scores_by_fake.tolist())

        scores_adv = np.array(scores)
        predictions_adv = (scores_adv >= genuine_threshold).astype(int)

        predictions_adv = predictions_adv.ravel().tolist()
        #print(predictions_adv)
        
        with open(os.path.join(output_dir, f'sample_probability_ambient_adversarial_SNR{snr_target_db}.txt'), 'w') as f:
            for filename, pred,score in zip(filenames_list, predictions_adv, scores):
                f.write(f"{filename} {score:.6f} {pred}\n")

    return 1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Ambient attack eval+gen')
    parser.add_argument('--snr', type=float, required=True, help="SNR target in dB for the adversarial samples")
    parser.add_argument('--rumore', type=str, required=True, help="Nome rumore")
    parser.add_argument('--model_type', type=str, required=True, choices=['full', 'light'], help='Tipo di modello da utilizzare: full o light')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt file')

    '''
    parser.add_argument('--use_fp16', action='store_true', default=True,
                    help='Use mixed precision for faster processing')
    parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model')
    parser.add_argument('--dataset_path', type=str, required=True,
                    help='Path to dataset')
    

    model_path = args.model_path
    dataset_path = args.dataset_path
    use__fp16 = args.use_fp16
    '''
    args = parser.parse_args()
    snr_target_db = args.snr
    model_type = args.model_type
    noise_dataset_path = "../ESC-50/train/"
    dataset_path = "../ASVspoof5"
    nome_rumore = args.rumore
    model_path = args.model
    
    print(f"Modello scelto {model_type}:", model_path)
    adversarial_sample_path = "./adversarial_samples_ambient_noise/snr{}".format(snr_target_db)
    os.makedirs(adversarial_sample_path, exist_ok=True)
    use_fp16 = True
    output_dir = f"./ambient_noise_probabilities/aasist_{model_type}/{nome_rumore}"
    result = generate_audio(model_path, use_fp16, dataset_path, output_dir, adversarial_sample_path, snr_target_db,nome_rumore, model_type)
    if result:
        print(f" Results saved in {output_dir}")