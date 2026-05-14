import torch
import os
import numpy as np




def taratura_dev_set(model_path, path_to_features, path_to_features1, output_dir, feat, feat_len, pad_chop, padding, model_type):
    
    output_dir = os.path.join(output_dir, model_type)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = torch.load(model_path, weights_only = False).to(device)
        model.eval()

    except Exception as e:
        print(f"   ERRORE CARICAMENTO MODELLO: {e}")
        import traceback
        traceback.print_exc()
        return
    
    
    asv_validation_set = ASVspoof2019("LA", path_to_features1, 'dev',
                                      feat, feat_len=feat_len, pad_chop=pad_chop, padding=padding)

    codec_validation_set = codecfake("LA", path_to_features, 'dev',
                                      feat, feat_len=feat_len, pad_chop=pad_chop, padding=padding)
    # Creazione dei DataLoader
    validation_set = ConcatDataset([codec_validation_set, asv_validation_set])


    valOriDataLoader = DataLoader(validation_set, batch_size=32,shuffle=False, num_workers=8))                                  

    all_scores_by_fake = []
    all_labels = []
    
    all_filenames = []
    
    total_samples = 0

    count = 0
   

    with torch.no_grad():
        #ip1_loader, idx_loader, score_loader = [], [], []
            
        for i in trange(len(valOriDataLoader), desc="Validation"):
            try:
                featOri, audio_fnOri, labelsOri = next(valOri_flow)
            except StopIteration:
                valOri_flow = iter(valOriDataLoader)
                featOri, audio_fnOri, labelsOri = next(valOri_flow)
                
            feat = reshape_features_for_model(featOri, args.model).to(args.device)
        
            feats, feat_outputs = feat_model(feat)

            if feat_outputs.shape[1] == 2:
                probs = torch.softmax(feat_outputs, dim=1)
                batch_scores_by_fake = probs[:, 1].cpu().numpy() # Normalizza i logit e prendi la probabilità della classe "real" e la salva in scores per ogni campione
            else:
                #scores = torch.sigmoid(outputs).cpu().numpy()
                batch_scores_by_fake = torch.sigmoid(feat_outputs).detach().cpu().numpy().reshape(-1)
        
            all_scores_by_fake.extend(batch_scores_by_fake)
            all_labels.extend(labelsOri.numpy())
            all_filenames.extend(audio_fnOri)
            total_samples += len(labelsOri)

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
        'Model Path': model_path,
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
               
    

if __main__ == "__main__":
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices = ['full', 'light'])
    
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='./features/codecfake_xls-r-5')
    parser.add_argument("-f1", "--path_to_features1", type=str, help="cotrain_dataset1_path",
                        default='./features/asvspoof_xls-r-5')

    parser.add_argument("-o", "--out_fold", type=str, help="output folder", 
                        required=False, default='./taratura_results')

    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r-5',
                        choices=["mel", "xls-r-5"])
    parser.add_argument("--feat_len", type=int, help="features length", default=50)
    parser.add_argument('--pad_chop', type=bool, nargs='?', const=True, default=False,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")

    result = taratura_dev_set(args.model_path, args.path_to_features, args.path_to_features1, args.out_fold, args.feat, args.feat_len, args.pad_chop, args.padding, args.model_type)
    if result == 1:
        print("Taratura dev set completata con successo. Risultati salvati in:", os.path.join(args.out_fold, args.model_type))
    
    
                        