import os
import numpy
import glob
import sys
import argparse
from sklearn.metrics import roc_curve, auc, confusion_matrix
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESC-50 Evaluation')
    parser.add_argument('--model_type', type=str, required=True, choices=["full","light"], help='full or light')
    args = parser.parse_args()
    model_type = args.model_type

    path_ambient_sample_prob =f"esc-50_probabilities/{model_type}/sample_probability_ESC-50.txt"
    output_dir = f"evaluation_results_ESC-50/aasist_{model_type}"
    
    
    os.makedirs(output_dir,exist_ok = True)

    all_labels = []
    
    with open(path_ambient_sample_prob,"r") as f:

        list_filenames_adv = []
        dic_filename_adv = {}
        list_predictions = []
        for line in f.readlines():
            filename, predicted = line.strip().split(" ")
            all_labels.append(0)
            list_filenames_adv.append(filename)
            dic_filename_adv[filename] = predicted
            list_predictions.append(int(predicted))
    
    # Inizializzo le metriche per questa combinazione di max_iter ed eps_step
    num_samples = len(list_filenames_adv)
    print(f"Numero campioni {num_samples}")
  
    # Correttamente classificati x accuracy
    correctly_classified = 0

    false_negative = 0

    cm = confusion_matrix(all_labels, list_predictions, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Quanti bonafide faccio passare
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            
             
    results = {
        
                'Accuracy (%)': accuracy * 100,
                'Precision (%)': precision * 100,
                'Recall (%)': recall * 100,
                'F1-Score': f1,
                'Specificity (%)': specificity * 100,
                'True Positives': tp,
                'True Negatives': tn,
                'False Positives': fp,
                'False Negatives': fn,
                'Total Samples': num_samples
            }

    # Salva risultati in file
    with open(os.path.join(output_dir, 'evaluation_ESC-50.txt'), 'w') as f:
        f.write("="*10 + "Model evaluation results" + "-" *10 + "\n")
              
                
        f.write("-"*10 + "PERFORMANCE METRICS:" + "-"*10 + "\n")
        for key, value in results.items():
            
                f.write(f"{key:<30}: {value}\n")
                
                f.write("\n") 
        
        f.write("-"*10 + "False Positive Samples:" + "-"*10 + "\n")
        for key, val in dic_filename_adv.items():
            if val == 1:
                f.write(key)
        
        
        