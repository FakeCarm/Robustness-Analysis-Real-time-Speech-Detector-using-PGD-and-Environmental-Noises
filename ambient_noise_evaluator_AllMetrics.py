import os
import numpy
import glob
import sys
import argparse
import eval_metrics as em
import numpy as np
import torch
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj
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

def evaluate(path_og_sample_prob,path_ambient_sample_prob, output_dir, snr, nome_amb_audio, model_type):

    os.makedirs(output_dir,exist_ok = True)

    labels = []
    # Prendo i risultati da tutti i campioni ottenuti nell'evaluation del test-set di asvspoof5
    with open(path_og_sample_prob,"r") as f:

        dic_filename_og = {}
        list_filenames_og = []
        for line in f.readlines():
            filename_extended ,prob , label, predicted = line.strip().split(" ")
            if model_type == "light":
                pre, name, attack, id_attack, l = filename_extended.split("_")
            else:
                pre, name, l = filename_extended.split("_")

            filename = pre + "_" + name
            dic_filename_og[filename] = (label,prob,predicted)
            list_filenames_og.append(filename)

    # Prendo i risultati solo dai campioni adversarial del test-set di asvspoof5
    with open(path_ambient_sample_prob,"r") as f:

        list_filenames_adv = []
        dic_filename_adv = {}
        for line in f.readlines():
            filename, prob, predicted = line.strip().split(" ")
            list_filenames_adv.append(filename)
            dic_filename_adv[filename] = (1,prob,predicted)
    
    # Inizializzo le metriche per questa combinazione di max_iter ed eps_step
    num_samples_adv = len(list_filenames_adv)
    num_samples_og = len(list_filenames_og)
    print(f"Numero campioni adversarial {num_samples_adv}")
    print(f"Numero campioni totali {num_samples_og}")
   
    
    
    # Correttamente misclassificati (originale predice fake ma adversarial bonafide)
    correctly_misclassified = 0
    # caso inverso: il campione era errato prima (predetto 0) e corretto dopo (1)
    misclassificazione_inversa = 0
    # numero di spoof originariamente classificati correttamente
    num_originally_correct_spoof = 0
    # Campioni originariamente errati (originale predice bonafide ma adversarial spoof)
    num_originally_wrong_spoof = 0

    accuracy = 0.0
    recall = 0.0
    precision = 0.0
    attack_success_rate = 0.0
    eer = 0.0
    minDCF = 0.0
    roc_auc = 0.0
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    missed = []
    lista_filenames_aggiornata = []
    lista_predizioni_aggiornata = []
    lista_label_aggiornata = []
    lista_scores_aggiornata = []
    # Imposto predicted og che viene riempito solo se sto prendendo un campione adversarial.
    # Mi serve per calcolare l'attack success rate
    predicted_og = 9999
    # Per ogni campioni
    for sample in list_filenames_og:
        '''
        if sample not in list_filenames_adv:
            label, predicted = dic_filename_og[sample]
        else:
            _, predicted_og = dic_filename_og[sample]
            label, predicted = dic_filename_adv[sample]
        '''
        label, score, predicted = dic_filename_og[sample]
        label = int(label)
        if label == 1:
            _, score, predicted_og = dic_filename_og[sample]
            label, score, predicted = dic_filename_adv[sample]

        label = int(label)
        predicted = int(predicted)
        predicted_og = int(predicted_og)
        score = float(score)
        
        lista_filenames_aggiornata.append(sample)
        lista_predizioni_aggiornata.append(predicted)
        lista_label_aggiornata.append(label)
        lista_scores_aggiornata.append(score)

        if predicted == label and label == 1:
            tp += 1
        elif predicted == label and label == 0:
            tn += 1
        elif predicted != label and label == 1:
            fn += 1
        elif predicted != label and label == 0:
            fp += 1
        
        

        if predicted_og != 9999:
            # numero di spoof originariamente classificati correttamente
            if label == 1 and predicted_og == 1:
                num_originally_correct_spoof += 1

            # vero successo dell'attacco
            if label == 1 and predicted_og == 1 and predicted == 0:
                correctly_misclassified += 1

            # caso inverso: il campione era errato prima e corretto dopo
            if label == 1 and predicted_og == 0 and predicted == 1:
                misclassificazione_inversa += 1
            
            if label == 1 and predicted_og == 0:
                num_originally_wrong_spoof += 1

        predicted_og = 9999  # numero di spoof originariamente classificati correttamente
        
    

    if len(lista_predizioni_aggiornata) == len(lista_label_aggiornata):
        print("Lunghezza liste uguali")
    else:
        return KeyError("ERRORE LUNGHEZZE LISTA")

    # 0. Calcolo metriche

    accuracy = (tp + tn)/(tp+tn+fn+fp) if len(lista_predizioni_aggiornata) > 0 else 0
    recall = (tp)/(tp + fn) if (tp + fn) > 0 else 0
    precision = (tp)/(tp + fp) if (tp + fp) > 0 else 0

    # 1. CALCOLA EER
    lista_scores_aggiornata_np = np.array(lista_scores_aggiornata)
    lista_label_aggiornata_np = np.array(lista_label_aggiornata)

    real_scores = lista_scores_aggiornata_np[lista_label_aggiornata_np == 0]
    fake_scores = lista_scores_aggiornata_np[lista_label_aggiornata_np == 1]
    # Utilizzo come target la classe 1 ovvero la classe spoof.
    eer, eer_thresholds, frr, far, min_index  = em.compute_eer(fake_scores,real_scores)

    # 2. Calcolo minDCF
    Cmiss = 1
    # costo di rifiutare un campione bonafide, cioè classificarlo come spoof

    Cfa = 10
    # costo di accettare un campione spoof, cioè classificarlo come bonafide

    Pspoof = 0.05
    # prior probability di spoof nello scenario applicativo

    # compute_eer(fake_scores, real_scores) restituisce:
    # - frr: spoof classificati come bonafide   (Dato un target, decido nontarget)
    # - far: bonafide classificati come spoof (Dato un Non target, decido target)

    FAR_cm = frr
    # spoof accettati come bonafide

    FRR_cm = far
    # bonafide rifiutati come spoof
    minDCF, _ = compute_mindcf(frr=FRR_cm, far=FAR_cm, thresholds=eer_thresholds, Pspoof=Pspoof, Cmiss=Cmiss, Cfa=Cfa)

    # 3. Calcolo AUC
    fpr, tpr, roc_thresholds = roc_curve(lista_label_aggiornata_np, lista_scores_aggiornata_np, pos_label=1)  
    roc_auc = auc(fpr, tpr)
    
    
    attack_success_rate = (
    correctly_misclassified / num_originally_correct_spoof * 100
    if num_originally_correct_spoof > 0 else 0.0
    )

    recovery_rate = (
    misclassificazione_inversa / num_originally_wrong_spoof * 100
    if num_originally_wrong_spoof > 0 else 0.0
    )
            
    print(f'Accuracy {accuracy}\n attack_success_rate {attack_success_rate}% \n')
    results = {
                'Audio ambientale':  nome_amb_audio,
                'snr': snr,
                'Threshold Modello Originale': 0.714,
                'EER (%)': eer * 100,
                'min_DCF': minDCF,
                'Accuracy (%)': accuracy * 100,
                'Recall (%)': recall * 100,
                'Precision (%)': precision * 100,
                '# Correctly Missclassified ': int(correctly_misclassified),
                'Accuracy Misclassified (%)': attack_success_rate,
                'Recovery Rate (%)': recovery_rate,
                '# Misclassified Recovered': int(misclassificazione_inversa),
                'Effetto Attacco (%)': (int(correctly_misclassified) - int(misclassificazione_inversa)) / num_samples_adv * 100 if num_samples_adv > 0 else 0.0,
                'FAR (%)': FAR_cm[min_index] * 100,
                'AUC': roc_auc,
                'Total Samples': num_samples_og,
                'Total adv samples': num_samples_adv,
                'True Positive': tp,
                'True Negative': tn,
                'False Positive': fp,
                'False Negative': fn
            }

    results = make_json_serializable(results)

    # Salva risultati in file
    with open(os.path.join(output_dir, f'evaluation_ambiental_noisy_snr{snr}.json'), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Metriche salvate in metriche_risultati.json")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ambient attack evaluation')
    parser.add_argument('--snr', type=float, required=True)
    parser.add_argument('--rumore', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=["full","light"], help='full or light')
    
    args = parser.parse_args()
    snr = args.snr
    nome_amb_audio = args.rumore
    model_type = args.model_type

    path_og_sample_prob = f"evaluation_results/{model_type}/sample_probability.txt"
    # path_ambient_sample_prob =f"ambient_noise_results_AASIST_FULL/{nome_amb_audio}/sample_probability_ambient_adversarial_SNR{snr}.txt"
    # Per valutare il light perchè al momento le cartelle hanno un nome diverso.
    path_ambient_sample_prob =f"ambient_noise_probabilities/aasist_{model_type}/{nome_amb_audio}/sample_probability_ambient_adversarial_SNR{snr}.txt"
    output_dir = f"evaluation_results_ambient_noise/aasist_{model_type}/{nome_amb_audio}"
    
    evaluate(path_og_sample_prob, path_ambient_sample_prob, output_dir, snr, nome_amb_audio, model_type)
    
    
            


        
        
    

            




