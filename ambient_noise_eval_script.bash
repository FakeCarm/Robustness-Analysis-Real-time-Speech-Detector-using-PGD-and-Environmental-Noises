#!/bin/bash

# Parametri fissi
MODEL="../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt"
MODEL_TYPE="full"

# Array dei parametri variabili (modificali con quelli che ti servono)
RUMORI=("bambini" "folla")
SNRS=("35" "25" "15" "10")

echo "Inizio esecuzione sequenziale..."

# Ciclo su ogni tipo di rumore
for rumore in "${RUMORI[@]}"; do
    # Ciclo su ogni valore di SNR
    for snr in "${SNRS[@]}"; do
        
        # Crea un nome per il file di output (es. prende le prime 3 lettere del rumore + snr.txt)
        # "vento" e "25" diventerà "ven25.txt"
        nome_corto=${rumore:0:3}
        output_file="${nome_corto}${snr}.txt"
        
        echo "=================================================="
        echo "Avvio: Rumore = $rumore | SNR = $snr"
        echo "Output log: $output_file"
        
        # ESECUZIONE SEQUENZIALE (Senza nohup e senza &)
        python ambient_noise_generator_new.py \
            --model "$MODEL" \
            --model_type "$MODEL_TYPE" \
            --rumore "$rumore" \
            --snr "$snr" > "$output_file"
            
        echo "✓ Completato: $rumore a SNR $snr"
        
    done
done

echo "=================================================="
echo "Tutte le esecuzioni sono terminate con successo!"