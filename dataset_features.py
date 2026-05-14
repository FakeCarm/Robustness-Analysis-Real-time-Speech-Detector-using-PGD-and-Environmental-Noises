import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate

class codecfake(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='xls', 
                 feat_len=50, pad_chop=True, padding='repeat', genuine_only=False):
        super(codecfake, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.feature = feature
        self.feat_len = feat_len
        #self.label = {"fake": 1, "real": 0}
        
        # Trova tutti i file .pt
        feature_dir = os.path.join(path_to_features, part, feature)
        self.all_files = []
        self.labels = []
        cont = 0
        if os.path.exists(feature_dir):
            for f in sorted(os.listdir(feature_dir)):
                '''if cont >= 1:
                    break'''
                if f.endswith('.pt'):
                    #cont += 1
                    self.all_files.append(os.path.join(feature_dir, f))
                    parts = f.replace('.pt', '').split('_')
                    label_str = parts[-1]
                    self.labels.append(int(label_str))
        
        print(f"Found {len(self.all_files)} feature files in {feature_dir}")
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        
        # Carica features
        featureTensor = torch.load(filepath)
        #print("featureTensor shape:", featureTensor.shape)
        # Parse filename
        parts = basename.replace('.pt', '').split('_')
        label_str = parts[-1]
        label = int(label_str)
        
        '''if 'crop' in basename:
            filename = '_'.join(parts[1:-2])
        else:
            filename = '_'.join(parts[1:-1])'''
        filename = '_'.join(parts[0:-2])
        
        return featureTensor, filename, label
    
    def collate_fn(self, samples):
        return default_collate(samples)

class ASVspoof5(Dataset):
    def __init__(self,path_to_features, part='train', feature='xls'):
        super(ASVspoof5,self).__init__()

        self.path_to_features = path_to_features
        self.part = part
        self.feature = feature
        self.label = {"spoof": 1, "bonafide": 0}

        self.all_files = []
        feature_dir = os.path.join(self.path_to_features, self.part, self.feature)
        self.labels = []
        if os.path.exists(feature_dir):
            for f in sorted(os.listdir(feature_dir)):
                if f.endswith('.pt'):
                    self.all_files.append(os.path.join(feature_dir, f))
                    # ESTRAI LA LABEL AL VOLO SENZA APRIRE IL FILE
                    parts = f.replace('.pt', '').split('_')
                    label_str = parts[-1]
                    self.labels.append(int(label_str))
        
        print(f"Found {len(self.all_files)} feature files in {feature_dir}")
        
    
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):

        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        
        # Carica features
        featureTensor = torch.load(filepath)
        
        # Parse filename
        #Formato  X_XXXXX_attack_XXX_label_xxx.pt
        parts = basename.replace('.pt', '').split('_')
        label_str = parts[-1]
        label = int(label_str)
        #label = self.label.get(label_str, 0) # Restituisce 1 se label_str è spoof, altrimenti 0 sia se bonafide che diverso da entrambi.
        id_attack = parts[-3] if len(parts) >= 3 else "unknown"
        
        if 'crop' in basename:
            filename = '_'.join(parts[0:-2]) # !!!!! CAMBIATO 1 con 0
        else:
            filename = '_'.join(parts[0:-1]) # !!!!! CAMBIATO 1 con 0
        
        # Ritorniamo il tensore, il nome del file, la label e l'id dell'attacco
        return featureTensor, filename, label

# Feat len rappresenta il numero di campioni di cui è composto un audio di 1 secondo a 16kHz, a 16kHz se ne prendono 50
class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='xls', feat_len=50, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
       
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        elif self.access_type == 'PA':
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        else:
            raise ValueError("Access type should be LA or PA!")
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            assert self.access_type == "LA"
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

        self.labels = []
        for filepath in self.all_files:
            basename = os.path.basename(filepath)
            parts = basename.replace('.pt', '').split('_')
            # Esattamente come facevi nel __getitem__
            label = int(parts[-1])
            self.labels.append(label)
        #self.all_files = self.all_files[0:100]
        print(f"Found {len(self.all_files)} feature files in {self.ptf}")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)

        #Formato LA_T_XXXXX_crop_x_label_x.pt
        # Carica features
        featureTensor = torch.load(filepath)
        #print("featureTensor shape:", featureTensor.shape)

        # "LA_T_1000137_crop_1_label_1.pt" diventa ['LA', 'T', '1000137', 'crop', '1', 'label', '1']
        # "LA_T_1000137_label_1.pt" diventa ['LA', 'T', '1000137', 'label', '1']
        parts = basename.replace('.pt', '').split('_')

        # La label è sempre l'ultimo elemento
        label = int(parts[-1])

        # Ricostruiamo il filename prendendo tutti gli elementi tranne gli ultimi due ("label" e il numero)
        # Così per i crop otterrai: "LA_T_1000137_crop_1"
        # E per quelli base otterrai: "LA_T_1000137"
        filename = '_'.join(parts[:-2])
        #Gestisci entrambi i formati di filename
        if "_crop" in basename:  
            tag = 0  # Default tag per file Codecfake/crop
        else:  
            # Eventuale logica per il tag ASVspoof originale
            tag = 0

        if self.feature == "wav2vec2_largeraw":
            #featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        '''if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')'''


        #featureTensor = featureTensor.squeeze(dim=0)
        #filename =  "_".join(all_info[1:4])
        #tag = self.tag[all_info[4]]
        #label = self.label[all_info[5]]
        
        return featureTensor, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)