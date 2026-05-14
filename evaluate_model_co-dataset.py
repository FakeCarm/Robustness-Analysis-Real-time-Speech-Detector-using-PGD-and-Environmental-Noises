import os    
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from dataset_features import ASVspoof5, codecfake, ASVspoof2019
from torch.utils.data import DataLoader
import eval_metrics as em
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from transformers import Wav2Vec2Model
import sys

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

def reshape_features_for_model(features, model_name):
    """Adatta le features al modello specifico"""
    # features arriva come [batch, seq_len, hidden_dim] = [64, 49, 1024]
    if model_name == 'W2VAASIST':
        # W2VAASIST vuole [batch, hidden_dim, seq_len] NON 4D!
        return features.transpose(1, 2)  # [64, 1024, 49]
    else:
        return features

def evaluate_model(model_path, dataset_path_codec,dataset_path_asv, output_dir, model_type):

    output_dir = os.path.join(output_dir, model_type)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path_codec = os.path.join(dataset_path_codec, model_type)
    dataset_path_asv = os.path.join(dataset_path_asv, model_type)
    

    try:
                
        ADD_MODEL = torch.load(model_path, weights_only = False).to(device)
        ADD_MODEL.eval()
    
    except Exception as e:
        print(f"   ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # dataset_features.py
    test_codecfake = codecfake("LA", dataset_path_codec, part='dev', feature='xls', feat_len = 50)
    test_asvspoof19 = ASVspoof2019("LA", dataset_path_asv, part='dev', feature='xls', feat_len = 50)
    if len(test_codecfake) == 0 and len(test_asvspoof19) == 0:
        raise RuntimeError(
            f"Nessun file .pt trovato. Controlla path: {os.path.abspath(dataset_path_codec)}  o"
            f"{os.path.abspath(dataset_path_asv)}"
            f"e la cartella eval/xls"
        )
    #test_dataset = torch.utils.data.ConcatDataset([test_codecfake, test_asvspoof19])
    test_loader = DataLoader(test_asvspoof19, batch_size=32, shuffle=False, num_workers=4)

    all_scores_by_fake = []
    all_labels = []
    
    all_filenames = []
    
    total_samples = 0

    feature_extraction_times = []
    inference_times = []

    count = 0
    #genuine_threshold = 0.7

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            features, filenames, labels = batch
            if count == 10:
                break
            #print("features batch shape:", features.shape)
            #import sys; sys.exit(0)
            '''if features.dim() == 3 and features.shape[-1] == 1024:
                features = features.transpose(1, 2).contiguous()'''
            features = reshape_features_for_model(features, "W2VAASIST").to(device)
            
            
            # Misura tempo estrazione features
            feat_start = time.time()
            # Se usi features pre-estratte, questo è già fatto
            feat_time = time.time() - feat_start
            feature_extraction_times.append(feat_time)
            
            # Misura tempo inferenza
            inf_start = time.time()
            _, outputs = ADD_MODEL(features)
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)
            
        # Calcola scores. 
            if outputs.shape[1] == 2:
            # Se il modello restituisce logit per 2 classi, applichiamo softmax e prendiamo la probabilità della classe "real" (index 0) e "fake" (index 1)
                probs = torch.softmax(outputs, dim=1)
                batch_scores_by_fake = probs[:, 1].cpu().numpy() # Normalizza i logit e prendi la probabilità della classe "real" e la salva in scores per ogni campione
            else:
                #scores = torch.sigmoid(outputs).cpu().numpy()
                batch_scores_by_fake = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
                
            
            
            all_scores_by_fake.extend(batch_scores_by_fake)
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)
            total_samples += len(labels)
            #count = count + 1
            
    
    # Converti a numpy arrays
    all_scores_by_fake = np.array(all_scores_by_fake)
    all_labels = np.array(all_labels)
    
    # 1. CALCOLA EER
    real_scores = all_scores_by_fake[all_labels == 0]
    fake_scores = all_scores_by_fake[all_labels == 1]
    # Utilizzo come target la classe 1 ovvero la classe spoof.
    eer, eer_thresholds, frr, far, min_index  = em.compute_eer(fake_scores,real_scores)
    eer_min_threshold = eer_thresholds[min_index]
    #Cmiss = 1 | Cfa = 10 da paper asvspoof5
    Cmiss = 1 # Costo misclassificazione falso negativo, viene etichettato un bonafide come spoof e quindi non viene accettato.
    Cfa = 10 # Costo misclassificazione falso positivo, viene accettato uno spoof come bonafide, e quindi viene accettato come buono un audio falso.
    
    # asserted prior probability of spoofing attack, in uno scenario reale in cui la stragrande maggioranza degli utenti sono legittimi, mentre solo lo 0.05 sono spoofing attack.
    # Non dipende dal dataset
    Pspoof = 0.05 
    FAR_cm = frr   # spoof accettati (score < t) Da P(decido real/spoof) a P(decido spoof/real)
    FRR_cm = far   # bonafide rifiutati (score >= t)
    minDCF, _ = compute_mindcf(frr=FRR_cm, far=FAR_cm, thresholds=eer_thresholds, Pspoof=Pspoof, Cmiss=Cmiss, Cfa=Cfa) 
    # 2. CALCOLA AUC e ROC Curve | accetta etichette, accetta scores per etichetta
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores_by_fake, pos_label=1)  
    roc_auc = auc(fpr, tpr)

    auc_real_as_score = roc_auc_score(all_labels, 1.0-all_scores_by_fake)          # con P(real)
    auc_fake_as_score = roc_auc_score(all_labels, all_scores_by_fake)    # se all_scores=P(real)
    print(auc_real_as_score, auc_fake_as_score)
    
    # 4. CALCOLA METRICHE AL THRESHOLD OTTIMALE
    predictions = (all_scores_by_fake >= eer_min_threshold).astype(int)
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Quanti bonafide faccio passare
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 5. METRICHE DI LATENZA
    avg_inference_time = np.mean(inference_times) * 1000  # in ms
    std_inference_time = np.std(inference_times) * 1000
    avg_feat_time = np.mean(feature_extraction_times) * 1000
    throughput = total_samples / sum(inference_times)  # samples/sec
    
    # 6. SALVA RISULTATI
    results = {
        'EER (%)': eer * 100,
        'minDCF': minDCF,
        'AUC': roc_auc,
        'EER Threshold': eer_min_threshold,
        'Accuracy (%)': accuracy * 100,
        'Precision (%)': precision * 100,
        'Recall (%)': recall * 100,
        'F1-Score': f1,
        'Specificity (%)': specificity * 100,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'Avg Inference Time (ms)': avg_inference_time,
        'Std Inference Time (ms)': std_inference_time,
        'Avg Feature Time (ms)': avg_feat_time,
        'Throughput (samples/sec)': throughput,
        'Total Samples': total_samples
        
    }

     # Salva risultati in file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*30 + "\n")
        for key, value in results.items():
            if 'Time' not in key and 'Throughput' not in key and 'Total' not in key:
                f.write(f"{key:<25}: {value:.3f}\n")
        
        f.write("\nLATENCY METRICS:\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Avg Inference Time':<25}: {avg_inference_time:.2f} ms\n")
        f.write(f"{'Std Inference Time':<25}: {std_inference_time:.2f} ms\n")
        f.write(f"{'Avg Feature Time':<25}: {avg_feat_time:.2f} ms\n")
        f.write(f"{'Throughput':<25}: {throughput:.1f} samples/sec\n")
        f.write(f"\n Total Samples: {total_samples}\n")
        
        f.write("\nCONFUSION MATRIX:\n")
        f.write("-"*30 + "\n")
        f.write(f"True Negatives:  {tn}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives:  {tp}\n")

    with open(os.path.join(output_dir, 'sample_probability.txt'), 'w') as f:
        for filename, label, score, pred in zip(all_filenames, all_labels, all_scores_by_fake, predictions):
            f.write(f"{filename} {score:.6f} {label} {pred}\n")
    return 1      

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--dataset_codec', type=str, required=True, help='Path to test data')
    parser.add_argument('--dataset_asv', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/')
    parser.add_argument('--benchmark', action='store_true', help='Run latency benchmark')
    parser.add_argument('--model_type', type=str, choices = ['full', 'light'], required=True)
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, args.model_type)
    # Valutazione completa
    results = evaluate_model(args.model, args.dataset_codec,args.dataset_asv, args.output_dir, args.model_type)
    if results == 1:
        print("Evaluation completed successfully. Results saved in:", args.output_dir)
