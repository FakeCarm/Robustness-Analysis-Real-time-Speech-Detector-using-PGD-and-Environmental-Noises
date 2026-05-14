import json
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- 4. FUNZIONE GENERAZIONE GRAFICO ---
def create_chart(metric_name, values_dict, current_noise_folder):
    y = np.arange(len(labels))
    height = 0.18
    # Usiamo layout='constrained' per gestire meglio gli spazi tra titolo e legenda
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained') 
    
    noise_names = list(noise_map.values())
    all_values = []

    for i, n_name in enumerate(noise_names):
        vals = [values_dict[n_name][s] for s in snr_levels]
        all_values.extend(vals)
        offset = (i - 1.5) * height
        rects = ax.barh(y - offset, vals, height, label=n_name, color=colors[i])
        
        for rect in rects:
            width = rect.get_width()
            # Logica etichette (nera se piccola/negativa, bianca se dentro la barra)
            if abs(width) < 5: 
                ha_val = 'left' if width >= 0 else 'right'
                text_color = 'black'
                offset_x = 5 if width >= 0 else -5
            else:
                ha_val = 'right' if width >= 0 else 'left'
                text_color = 'white'
                offset_x = -5 if width >= 0 else 5

            ax.annotate(f'{width:.2f}%',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(offset_x, 0), textcoords="offset points",
                        ha=ha_val, va='center', color=text_color, 
                        fontweight='bold', fontsize=9)

    # --- TITOLO CON PAD MAGGIORE ---
    ax.set_title(f'{metric_name} (%)\nAASIST-{model_type} for each noise at a given dB', 
                 fontsize=15, fontweight='bold', pad=30)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    
    # --- GESTIONE ASSI ---
    min_val = min(all_values)
    max_val = max(all_values)
    ax.set_xlim(min(min_val - 5, 0), max(max_val + 10, 100))
    
    if min_val < 0:
        ax.axvline(0, color='black', linewidth=1, zorder=3)
        
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    # --- LEGENDA SOTTO IL GRAFICO (Soluzione all'overlap) ---
    ax.legend(loc='upper center', 
              bbox_to_anchor=(0.5, -0.08), # Posizionata sotto l'asse X
              ncol=4, 
              frameon=False, 
              fontsize=11)
    
    # Salvataggio
    final_dir = os.path.join(output_path, current_noise_folder)
    os.makedirs(final_dir, exist_ok=True)
    filename = f"plot_{model_type}_{metric_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(final_dir, filename), dpi=300)
    plt.close()

# --- 1. CONFIGURAZIONE PERCORSI ---
base_path = '/home/carmine/Scrivania/Carmine_P_R_Cavaliere_Code/main/evaluation_results_ambient_noise'
model_type = 'light'
target_folder_base = os.path.join(base_path, "aasist_"+model_type)

output_path = f"graph/{model_type}"
os.makedirs(output_path, exist_ok=True)

# --- 2. MAPPATURE ---
noise_map = {
    "bambini": "Children",
    "vento": "Wind",
    "folla": "Crowd",
    "pioggia": "Rain"
}

metrics_to_plot = {
    "EER": "EER (%)",
    "Recovery Rate": "Recovery Rate (%)",
    "Attack Effect": "Effetto Attacco (%)",
    "Attack Success Rate": "Accuracy Misclassified (%)"
}

snr_levels = [10.0, 15.0, 25.0, 35.0]
labels = [f"{int(s)}dB" for s in snr_levels]
colors = ['#4F81FF', '#B08FFF', '#F5B96D', '#FFE162']

# Inizializzazione struttura dati
data = {m: {n: {s: 0.0 for s in snr_levels} for n in noise_map.values()} for m in metrics_to_plot}
noises = ["bambini", "vento", "folla", "pioggia"]

# --- 3. ESTRAZIONE DATI ---
for n in noises:
    current_target_folder = os.path.join(target_folder_base, n)
    search_pattern = os.path.join(current_target_folder, "*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"ATTENZIONE: Nessun file JSON trovato in {current_target_folder}")
        continue

    print(f"Elaborazione {n}...")
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                noise_type = content.get("Audio ambientale")
                snr_val = content.get("snr")
                
                if noise_type in noise_map and snr_val in snr_levels:
                    eng_noise = noise_map[noise_type]
                    for metric_name, json_key in metrics_to_plot.items():
                        val = content.get(json_key, 0.0)
                        data[metric_name][eng_noise][snr_val] = val
        except Exception as e:
            print(f"Errore nel file {file_path}: {e}")

# --- 5. GENERAZIONE GRAFICI (Fuori dal loop dei file) ---
# Generiamo i grafici una volta sola dopo aver letto tutti i file
for metric in metrics_to_plot:
    # Passo 'all_noises' come nome cartella o gestisci come preferisci
    create_chart(metric, data[metric], "summary") 

print("Operazione Completata")