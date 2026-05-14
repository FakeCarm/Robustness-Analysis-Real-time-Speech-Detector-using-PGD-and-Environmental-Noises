#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
from typing import Tuple
import soundfile as sf

torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


class codecfake_eval(Dataset):
    def __init__(self, type):
        super(codecfake_eval, self).__init__()
        self.type = type
        
        # PERCORSI CORRETTI
        base_paths = [
            './Codecfake/test',
            './Codecfake',
            '../Codecfake/test',
            '/user/nlanzara/Codecfake/test',
        ]
        
        # Trova il path corretto
        self.path_to_audio = None
        self.path_to_protocol = None
        
        for base_path in base_paths:
            audio_path = os.path.join(base_path, self.type)
            protocol_path = os.path.join(base_path, 'label', self.type + '.txt')
            
            if os.path.exists(protocol_path) and os.path.exists(audio_path):
                self.path_to_audio = audio_path
                self.path_to_protocol = protocol_path
                print(f"✅ Found {self.type} data at: {base_path}")
                break
        
        if self.path_to_protocol is None:
            self.path_to_audio = f'./Codecfake/test/{self.type}'
            self.path_to_protocol = f'./Codecfake/label/{self.type}.txt'
            print(f"⚠️ Using default paths for {self.type}")
        
        # Carica il protocollo
        self.all_info = []
        skipped = 0
        try:
            with open(self.path_to_protocol, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename, label = parts[0], parts[1]
                        filepath = os.path.join(self.path_to_audio, filename)

                        #Verifica che il file esista PRIMA di aggiungerlo
                        if os.path.exists(filepath):
                            self.all_info.append((filename, label))
                        else:
                            skipped += 1
            
            print(f"📊 Loaded {len(self.all_info)} samples, skipped {skipped} missing files")
            
        except FileNotFoundError:
            print(f"❌ Protocol file not found: {self.path_to_protocol}")
            self.all_info = []

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        if idx >= len(self.all_info):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.all_info)}")
        
        # Ora abbiamo sempre esattamente 2 valori
        filename, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        
        try:
            waveform, sr = torchaudio_load(filepath)
            # RITORNA SEMPRE 3 VALORI: waveform, filename, label
            return waveform, filename, label
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return dummy data con 3 valori
            return torch.zeros(1, 16000), filename, label

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2019LAeval(Dataset):
    def __init__(self):
        super(ASVspoof2019LAeval, self).__init__()
        
        base_paths = [
            './ASVspoof2019/LA',
            '../ASVspoof2019/LA',
            '/user/nlanzara/ASVspoof2019/LA'
        ]
        
        for base_path in base_paths:
            audio_path = os.path.join(base_path, 'ASVspoof2019_LA_eval/flac')
            protocol_path = os.path.join(base_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
            
            if os.path.exists(protocol_path):
                self.path_to_audio = audio_path
                self.path_to_protocol = protocol_path
                print(f"Found ASVspoof2019 eval data at: {base_path}")
                break
        else:
            print("WARNING: Could not find ASVspoof2019 eval data!")
            self.all_info = []
            return
        
        # Processo il file protocol
        self.all_info = []
        with open(self.path_to_protocol, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, filename, _, _, label = parts
                    # Salva solo filename e label per consistency
                    self.all_info.append((filename, label))

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + '.flac')
        
        try:
            waveform, sr = torchaudio_load(filepath)
            # RITORNA SEMPRE 3 VALORI
            return waveform, filename, label
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, 16000), filename, label

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2019Raw(Dataset):
    def __init__(self, access_type, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019Raw, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.path_to_protocol = path_to_protocol
        
        if self.part =='train':
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trn.txt')
        else:
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        
        self.all_info = []
        with open(protocol, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker, filename, _, tag, label = parts
                    label = 0 if label.lower() == 'bonafide' else 1
                    self.all_info.append((filename, label))
        print(f"Loaded ASVspoof2019Raw {self.access_type} {self.part} dataset with {len(self.all_info)} samples")

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + ".flac")
        
        try:
            waveform, sr = torchaudio_load(filepath)
            return waveform, filename, label
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, 16000), filename, label

    def collate_fn(self, samples):
        return default_collate(samples)


class codecfake(Dataset):   #Produce una lista di coppie (filename, label) per ogni file audio presente nella cartella specificata dal protocollo. Il dataset è progettato per essere utilizzato con il protocollo di Codecfake, che specifica quali file audio devono essere inclusi e le loro etichette corrispondenti (ad esempio, "bonafide" o "spoof"). Il dataset carica solo i file audio che esistono effettivamente nel percorso specificato, evitando errori di file mancanti durante l'addestramento o la valutazione del modello.
    def __init__(self, path_to_database, path_to_protocol, part='train'):
        super(codecfake, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, self.part)
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(self.path_to_protocol, self.part + '.txt')
        
        self.all_info = []
        total = 0
        
        print(f"Loading {part} dataset from {protocol}")
        with open(protocol, 'r') as f:
            for line in f:
                total += 1
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = parts[1]
                    filepath = os.path.join(self.path_to_audio, filename)
                    label = 0 if label.lower() == 'real' else 1
                    if os.path.exists(filepath):
                        self.all_info.append((filename, label))
        
        print(f"Dataset {part}: Found {len(self.all_info)}/{total} files")

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        
        try:
            waveform, sr = torchaudio_load(filepath)
            return waveform, filename, label
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, 16000), filename, label

    # Funzione collate_fn per gestire i batch di dati, utile quando si utilizza un DataLoader. Questa funzione prende una lista di campioni (ognuno dei quali è una tupla contenente il waveform, il filename e l'etichetta) e li combina in un batch utilizzando la funzione default_collate di PyTorch, che gestisce automaticamente la creazione di tensori batch a partire da liste di campioni.
    # Si utilizza per gestire in modo custom i campioni durante il caricamento dei dati, ad esempio per gestire casi in cui i file audio potrebbero essere di lunghezze diverse o per applicare trasformazioni specifiche ai campioni prima di formarli in batch.
    def collate_fn(self, samples):
        return default_collate(samples)
    
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

        with open(protocol, 'r') as f:
            lines = f.readlines()
            
        
        for line in lines:
            total += 1
            parts = line.strip().split(' ')
            if len(parts) == 10:
                filename = parts[1]
                label = parts[8]
                id_attack = parts[7]
                filepath = os.path.join(self._path_to_audio, filename + self._formato)
                    
                if os.path.exists(filepath):
                    self.all_info.append((filename, label, id_attack))


        print(f"Dataset ASVspoof5 {part}: Found {len(self.all_info)}/{total} files")
    
    def __len__(self):
        return len(self.all_info)
    
    def __getitem__(self, idx):
        filename, label, id_attack = self.all_info[idx]
        filepath = os.path.join(self._path_to_audio, filename + self._formato)
        #print(f"Loading file: {filepath} (ID Attack: {id_attack}, Label: {label})")
        try:
            waveform, sr = torchaudio_load(filepath)
            return waveform, filename, label
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros(1, 16000), filename, label



