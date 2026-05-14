# Anti-Spoofing Audio Detection — Documentazione del Progetto

Questo progetto implementa un sistema di **rilevamento audio sintetico (anti-spoofing)** basato su modelli neurali (AASIST + Wav2Vec2). Il sistema è in grado di distinguere audio reale (*bonafide*) da audio generato artificialmente (*spoof*). Il progetto include anche la valutazione della robustezza del modello rispetto ad **attacchi avversariali** (PGD) e a **rumore ambientale** (ambient noise).

---

## Indice

1. [Architettura Generale](#1-architettura-generale)
2. [Requisiti e Dipendenze](#2-requisiti-e-dipendenze)
3. [Struttura delle Cartelle Attesa](#3-struttura-delle-cartelle-attesa)
4. [Descrizione dei File](#4-descrizione-dei-file)
5. [Pipeline Completa di Esecuzione](#5-pipeline-completa-di-esecuzione)
6. [Output Prodotti da Ogni Script](#6-output-prodotti-da-ogni-script)
7. [Glossario delle Metriche](#7-glossario-delle-metriche)

---

## 1. Architettura Generale

Il sistema è composto da due componenti principali in cascata:

```
Audio grezzo (.flac/.wav)
        │
        ▼
[ Wav2Vec2-XLS-R-300M ]   ← Estrattore di features (modello pre-addestrato)
        │
        ▼
[ AASIST Classifier ]     ← Classificatore bonafide / spoof
        │
        ▼
  Score di probabilità    (0.0 = bonafide, 1.0 = spoof)
        │
        ▼
  Decisione con soglia    (default: 0.714 per modello "full", 0.7 per "light")
```

Esistono due varianti del modello:
- **`full`**: usa tutti i 24 layer transformer di Wav2Vec2.
- **`light`**: usa solo i primi 5 layer transformer di Wav2Vec2, più veloce ma leggermente meno accurato.

---

## 2. Requisiti e Dipendenze

```bash
pip install torch torchaudio transformers librosa scikit-learn numpy tqdm matplotlib adversarial-robustness-toolbox
```

Librerie principali:
- `torch`, `torchaudio` — deep learning e audio processing
- `transformers` — caricamento di Wav2Vec2 da HuggingFace
- `librosa` — caricamento e manipolazione audio
- `scikit-learn` — metriche di valutazione (EER, AUC, confusion matrix)
- `art` (Adversarial Robustness Toolbox) — attacchi PGD avversariali
- `matplotlib` — generazione grafici

---

## 3. Struttura delle Cartelle Attesa

```
progetto/
│
├── main_train.py
├── extract_features_optimized_og.py
├── evaluate_model_asvspoof5.py
├── evaluate_model_co-dataset.py
├── ambient_noise_generator_new.py
├── ambient_noise_evaluator_AllMetrics.py
├── ambient_noise_eval_script.bash
├── adversarial_sample_generation.py
├── adversarial_evaluator_AllMetrics.py
├── esc-50-aasist-eval.py
├── generate_graph.py
├── dataset_features.py
├── raw_dataset.py
├── eval_metrics.py
├── taratura_dev_set.py
│
├── features/                     ← Feature pre-estratte (.pt)
│   ├── codecfake/full/
│   ├── codecfake/ligh/
│   ├── asvspoof2019/full/
│   ├── asvspoof2019/light/
│   └── asvspoof5/full/
│   └── asvspoof5/light/

│
├── evaluation_results/           ← Risultati valutazione sul test set
├── evaluation_results_ambient_noise/
├── evaluation_results_adversarial_robustness/
├── ambient_noise_probabilities/
├── adversarial_results/
│
├── ../ASVspoof5/                 ← Dataset ASVspoof5 (audio grezzi)
├── ../ASVspoof19/LA/             ← Dataset ASVspoof2019
├── ../Codecfake/                 ← Dataset Codecfake
├── ../ESC-50/train/              ← Dataset ESC-50 (file audio di rumore ambientale)
└── ../Models/cotrain_AASIST_FULL/ ← Modello addestrato (.pt)
```

---

## 4. Descrizione dei File

---

### `raw_dataset.py`

**A cosa serve:** Definisce le classi `Dataset` di PyTorch per il caricamento degli **audio grezzi** (file `.flac` e `.wav`), senza pre-elaborazione in features. Viene usato principalmente da `extract_features_optimized_og.py` per caricare l'audio prima di estrarne le features con Wav2Vec2.

**Cosa fa:** Contiene le seguenti classi Dataset:
- `ASVspoof2019Raw` — carica i file audio del dataset ASVspoof2019 direttamente da disco.
- `ASVspoof5` — carica solo i campioni *spoof* del dataset ASVspoof5.
- `codecfake` — carica i file audio del dataset Codecfake leggendo il relativo protocollo (file `.txt` che specifica i filename e le label).
- `codecfake_eval` — versione per la sola fase di valutazione del dataset Codecfake.
- `ASVspoof2019LAeval` — versione per la sola fase di valutazione del dataset ASVspoof2019 LA.

Ogni classe:
1. Legge il file di protocollo (`.txt` o `.tsv`) per sapere quali file caricare e quali label associare.
2. Verifica che i file esistano prima di aggiungerli alla lista.
3. Implementa `__getitem__` che carica il waveform a 16kHz in mono e restituisce la tupla `(waveform, filename, label)`.

**Come si avvia:** Non viene avviato direttamente. È un modulo importato da altri script.

---

### `extract_features_optimized_og.py`

**A cosa serve:** Esegue la **pre-estrazione delle features** audio usando il modello Wav2Vec2-XLS-R-300M. Questo passaggio trasforma i file audio grezzi in tensori di feature (`.pt`) salvati su disco, che verranno poi usati dall'addestratore e dai valutatori senza dover rieseguire l'estrattore ogni volta. Si tratta di un passaggio obbligatorio da fare **prima** dell'addestramento.

**Cosa fa:**
1. Carica il modello Wav2Vec2 da HuggingFace (`facebook/wav2vec2-xls-r-300m`).
2. Per ogni file audio del dataset scelto, lo carica a 16kHz, lo normalizza e lo invia al modello.
3. Estrae gli hidden states del layer desiderato e li salva come file `.pt` nella cartella di output.
4. Se `--skip_existing True`, salta i file già elaborati per riprendere un'estrazione interrotta.
5. Supporta la modalità `full` (tutti i 24 layer) e `light` (solo i primi 5 layer).

**Come si avvia:**

```bash
# Estrazione per Codecfake (modello full)
python extract_features_optimized_og.py \
    --dataset codecfake \
    --dataset_path ../Codecfake/upload_zenodo \
    --output_path features \
    --model_type full \
    --skip_existing False

# Estrazione per ASVspoof2019
python extract_features_optimized_og.py \
    --dataset asvspoof2019 \
    --dataset_path ../ASVspoof19/LA \
    --output_path features \
    --model_type full \
    --skip_existing False

# Estrazione per ASVspoof5
python extract_features_optimized_og.py \
    --dataset asvspoof5 \
    --dataset_path ../ASVspoof5 \
    --output_path features \
    --model_type full \
    --skip_existing False
```

**Produce:** File `.pt` nelle cartelle `features/<dataset>/<model_type>/<split>/xls/`, uno per ogni campione audio. Il nome del file include il filename originale e la label (es. `LA_T_1000137_label_1.pt`).

---

### `dataset_features.py`

**A cosa serve:** Definisce le classi `Dataset` di PyTorch per il caricamento delle **features pre-estratte** (file `.pt`). Viene usato da `main_train.py`, `evaluate_model_asvspoof5.py` e `evaluate_model_co-dataset.py` al posto di `raw_dataset.py`, poiché le features sono già pronte su disco.

**Cosa fa:** Contiene tre classi Dataset:
- `ASVspoof2019` — carica le features pre-estratte di ASVspoof2019 dalla cartella `features/asvspoof2019/`.
- `ASVspoof5` — carica le features pre-estratte di ASVspoof5 dalla cartella `features/asvspoof5/`.
- `codecfake` — carica le features pre-estratte di Codecfake dalla cartella `features/codecfake/`.

Ogni classe:
1. Elenca tutti i file `.pt` nella cartella corrispondente a dataset + split + tipo feature.
2. Estrae la label dal nome del file (l'ultimo campo prima di `.pt` separato da `_`).
3. Implementa `__getitem__` che carica il tensore di feature con `torch.load` e restituisce `(featureTensor, filename, label)`.

**Come si avvia:** Non viene avviato direttamente. È un modulo importato da altri script.

---

### `main_train.py`

**A cosa serve:** È lo script principale di **addestramento** del classificatore AASIST. Addestra il modello a distinguere audio bonafide da audio spoof usando le features pre-estratte da Wav2Vec2.

**Cosa fa:**
1. Legge i parametri da linea di comando (dataset, epochs, batch size, learning rate, ecc.).
2. Carica i dataset di training e validation dalla cartella `features/` usando `dataset_features.py`.
3. Se `--train_task co-train`, addestra il modello su **entrambi** i dataset Codecfake e ASVspoof2019 contemporaneamente (co-training).
4. Se `--CSAM True`, applica la tecnica CSAM (Class-Specific Attention Module) durante il training.
5. Se `--use_weighted_sampler True`, usa un sampler pesato per bilanciare le classi.
6. A ogni epoca, calcola le metriche di validation (EER, minDCF) e salva il miglior modello.
7. Applica una loss pesata per gestire lo sbilanciamento tra classi bonafide e spoof.

**Come si avvia:**

```bash
# Co-training su Codecfake + ASVspoof2019
python main_train.py \
    --path_to_features features/codecfake/full \
    --path_to_features1 features/asvspoof2019/full \
    --out_fold ../Models/cotrain_AASIST_FULL \
    --train_task co-train \
    --CSAM True \
    --feat xls \
    --use_weighted_sampler True
```

**Produce:** Nella cartella `--out_fold`:
- `anti-spoofing_feat_model.pt` — il modello migliore salvato (usato da tutti gli altri script).
- `training_log.txt` — log con le metriche di ogni epoca.

---

### `evaluate_model_asvspoof5.py`

**A cosa serve:** Valuta le **prestazioni baseline** del modello addestrato sul test set di **ASVspoof5**, usando le features pre-estratte. Questo è il punto di partenza prima di qualunque analisi di robustezza.

**Cosa fa:**
1. Carica il modello salvato (`.pt`) e il dataset ASVspoof5 dalla cartella `features/`.
2. Esegue l'inferenza su tutti i campioni del test set.
3. Calcola le metriche: EER, minDCF, AUC, Accuracy, Precision, Recall, F1, Specificity, latenza.
4. Salva le predizioni per ogni campione in un file di testo per le analisi successive.

**Come si avvia:**

```bash
python evaluate_model_asvspoof5.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --data features/asvspoof5 \
    --output_dir evaluation_results \
    --model_type full
```

**Produce:** Nella cartella `evaluation_results/<model_type>/`:
- `metrics.txt` — tutte le metriche di valutazione e latenza.
- `sample_probability.txt` — riga per riga: `<filename> <score> <label_vera> <predizione>`. **Questo file è fondamentale:** viene usato come input da `ambient_noise_evaluator_AllMetrics.py` e `adversarial_evaluator_AllMetrics.py`.

---

### `evaluate_model_co-dataset.py`

**A cosa serve:** Valuta le **prestazioni del modello sul dev set** dei dataset usati per il co-training (ASVspoof2019 e/o Codecfake), al fine di misurare la generalizzazione su dati mai visti durante il test.

**Cosa fa:** Identica a `evaluate_model_asvspoof5.py` ma carica il dataset ASVspoof2019 (o Codecfake) dalla cartella features e valuta sul split `dev`.

**Come si avvia:**

```bash
python evaluate_model_co-dataset.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --dataset_asv features/asvspoof2019 \
    --dataset_codec features/codecfake \
    --output_dir evaluation_results_codatasets \
    --model_type full
```

**Produce:** Nella cartella `evaluation_results_codatasets/<model_type>/`:
- `metrics.txt` — metriche di valutazione sul dev set.
- `sample_probability.txt` — predizioni per ogni campione del dev set.

---

### `taratura_dev_set.py`

**A cosa serve:** Script di **taratura** (calibrazione) del modello sul dev set combinato di Codecfake e ASVspoof2019. Serve a trovare la soglia di decisione ottimale (threshold EER) da usare poi nel test set.

**Cosa fa:** Funziona in modo quasi identico a `evaluate_model_co-dataset.py`, ma:
1. Combina i due dataset di validation in un unico `ConcatDataset`.
2. Calcola la soglia EER ottimale sul dataset combinato.
3. Salva le probabilità per ogni campione.

> ⚠️ **Nota:** Il file contiene alcuni bug minori (es. `if __main__ == "__main__"` invece di `if __name__ == "__main__"`) che vanno corretti prima dell'esecuzione.

**Come si avvia:**

```bash
python taratura_dev_set.py \
    --model_path ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full \
    --path_to_features features/codecfake/full \
    --path_to_features1 features/asvspoof2019/full \
    --out_fold ./taratura_results
```

**Produce:** Nella cartella `taratura_results/<model_type>/`:
- `metrics.txt` — metriche e soglia EER calcolata sul dev set.
- `sample_probability.txt` — predizioni per ogni campione.

---

### `eval_metrics.py`

**A cosa serve:** Modulo di **utilità per il calcolo delle metriche** di valutazione audio anti-spoofing. Non va eseguito direttamente: viene importato dagli altri script.

**Cosa fa:** Contiene le seguenti funzioni:
- `compute_det_curve(target_scores, nontarget_scores)` — calcola la curva DET (False Rejection Rate vs False Acceptance Rate) al variare della soglia.
- `compute_eer(target_scores, nontarget_scores)` — calcola l'**Equal Error Rate** (EER), ovvero il punto in cui FRR = FAR. Restituisce anche i vettori frr, far e le soglie.
- `compute_tDCF(...)` — calcola la **tandem Detection Cost Function** (t-DCF), metrica usata in ASVspoof 2019.
- `obtain_asv_error_rates(...)` — calcola le error rates per il sistema ASV (Automatic Speaker Verification).

**Come si avvia:** Non viene avviato direttamente. Si importa con `import eval_metrics as em`.

---

### `ambient_noise_generator_new.py`

**A cosa serve:** Simula un **attacco con rumore ambientale realistico** mescolando audio del dataset ESC-50 (vento, pioggia, folla, bambini) con i campioni audio di ASVspoof5, a un rapporto segnale/rumore (SNR) controllato. Valuta poi come il rumore influenza le predizioni del modello.

**Cosa fa:**
1. Carica il modello Wav2Vec2 + AASIST dalla pipeline di inferenza.
2. Carica un file audio di rumore dal dataset ESC-50 (es. `pioggia.wav`).
3. Per ogni campione *spoof* del test set di ASVspoof5, mescola il rumore con l'audio originale al target SNR specificato.
4. Esegue l'inferenza del modello sull'audio con rumore.
5. Salva le predizioni (score e label predetta) in un file di testo.

**Come si avvia:**

```bash
# Singola esecuzione
python ambient_noise_generator_new.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full \
    --rumore pioggia \
    --snr 10

# Oppure in background con log
nohup python ambient_noise_generator_new.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full \
    --rumore pioggia \
    --snr 10 > pio10.txt &
```

Valori validi per `--rumore`: `pioggia`, `vento`, `folla`, `bambini` (devono corrispondere ai file `.wav` in `../ESC-50/train/`).

**Produce:** Nella cartella `ambient_noise_probabilities/aasist_<model_type>/<rumore>/`:
- `sample_probability_ambient_adversarial_SNR<snr>.txt` — riga per riga: `<filename> <score> <predizione>`.

---

### `ambient_noise_eval_script.bash`

**A cosa serve:** Script bash di **automazione sequenziale** per eseguire `ambient_noise_generator_new.py` su tutte le combinazioni di tipo di rumore e livello SNR, senza parallelismo (per evitare conflitti di risorse GPU).

**Cosa fa:**
1. Definisce gli array di rumori (`bambini`, `folla`) e livelli SNR (`35`, `25`, `15`, `10`).
2. Esegue in sequenza `ambient_noise_generator_new.py` per ogni combinazione.
3. Salva il log di ogni esecuzione in un file separato (es. `bam35.txt`, `fol10.txt`).

**Come si avvia:**

```bash
# Rendi eseguibile (solo la prima volta)
chmod +x ambient_noise_eval_script.bash

# Avvio in background con log
nohup ./ambient_noise_eval_script.bash \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full > ambfull.txt &
```

> **Nota:** I parametri `--model` e `--model_type` nella sintassi bash sono definiti come variabili interne allo script; modifica direttamente `MODEL` e `MODEL_TYPE` nel file per cambiare i valori predefiniti.

**Produce:** Tutti i file `sample_probability_ambient_adversarial_SNR<snr>.txt` per ogni combinazione rumore/SNR.

---

### `ambient_noise_evaluator_AllMetrics.py`

**A cosa serve:** Calcola tutte le **metriche di robustezza** per l'esperimento di rumore ambientale, confrontando le predizioni originali del modello (senza rumore) con quelle ottenute con il rumore aggiunto.

**Cosa fa:**
1. Legge le predizioni baseline da `evaluation_results/<model_type>/sample_probability.txt`.
2. Legge le predizioni rumorose da `ambient_noise_probabilities/aasist_<model_type>/<rumore>/sample_probability_ambient_adversarial_SNR<snr>.txt`.
3. Per ogni campione *spoof*: sostituisce lo score originale con quello ottenuto con il rumore.
4. Calcola le seguenti metriche:
   - **EER** (Equal Error Rate)
   - **minDCF** (minimum Detection Cost Function)
   - **AUC** (Area Under the ROC Curve)
   - **Accuracy**, **Recall**, **Precision**
   - **Attack Success Rate** — percentuale di campioni spoof originariamente classificati correttamente che ora vengono classificati come bonafide (successo del "disturbo").
   - **Recovery Rate** — percentuale di campioni spoof originariamente classificati male che ora vengono classificati correttamente (effetto correttivo del rumore).
   - **Effetto Attacco** — differenza netta tra misclassificazioni corrette e inverse.
   - **TP, TN, FP, FN** — valori della confusion matrix.

**Come si avvia:**

```bash
python ambient_noise_evaluator_AllMetrics.py \
    --snr 15 \
    --rumore pioggia \
    --model_type full
```

**Produce:** Nella cartella `evaluation_results_ambient_noise/aasist_<model_type>/<rumore>/`:
- `evaluation_ambiental_noisy_snr<snr>.json` — file JSON con tutte le metriche per quella combinazione rumore/SNR.

---

### `adversarial_sample_generation.py`

**A cosa serve:** Genera **campioni audio avversariali** usando l'attacco **PGD (Projected Gradient Descent)** tramite la libreria ART (Adversarial Robustness Toolbox). L'obiettivo è modificare l'audio spoof nel modo minimo possibile (perturbazione impercettibile) per far sì che il modello lo classifichi erroneamente come bonafide.

**Cosa fa:**
1. Carica il modello completo (Wav2Vec2 + AASIST) come pipeline differenziabile.
2. Avvolge la pipeline in un `PyTorchClassifier` di ART.
3. Per ogni campione spoof del test set, applica l'attacco PGD con diverse combinazioni di `max_iter` (numero di iterazioni) e `eps_step` (dimensione del passo di perturbazione).
4. Esegue l'inferenza del modello sull'audio perturbato.
5. Calcola metriche come: accuratezza post-attacco, attack success rate, SNR della perturbazione.
6. Salva le predizioni e le metriche.

**Come si avvia:**

> ⚠️ In questo script il parser degli argomenti è commentato; i path del modello e del dataset sono definiti direttamente nelle ultime righe di `__main__`. Prima dell'uso, modifica le variabili `model_path` e `dataset_path`.

```bash
python adversarial_sample_generation.py
```

**Produce:** Nella cartella `adversarial_results/`:
- `sample_probability_adversarial_MaxIters<n>_EpsStep<e>.txt` — predizioni per ogni campione e ogni combinazione di parametri.
- `metrics_<n>_<e>.txt` — metriche di performance per ogni combinazione.

---

### `adversarial_evaluator_AllMetrics.py`

**A cosa serve:** Calcola tutte le **metriche di robustezza avversariale** confrontando le predizioni baseline con quelle sui campioni perturbati con PGD. Funziona in modo analogo a `ambient_noise_evaluator_AllMetrics.py` ma per gli attacchi avversariali.

**Cosa fa:**
1. Legge le predizioni baseline da `evaluation_results/sample_probability.txt`.
2. Legge le predizioni avversariali da `adversarial_results/sample_probability_adversarial_MaxIter<n>_EpsStep<e>.txt`.
3. Calcola le stesse metriche di `ambient_noise_evaluator_AllMetrics.py` (EER, minDCF, AUC, Attack Success Rate, Recovery Rate, ecc.).
4. Supporta la modalità **trasferibilità** (`--trasf True`): valuta se gli esempi avversariali generati su RawNet riescono a ingannare AASIST (attacco black-box transfer).

**Come si avvia:**

```bash
# Analisi di robustezza (attacco sullo stesso modello)
python adversarial_evaluator_AllMetrics.py \
    --model_type full \
    --iter 10 \
    --eps 0.01 \
    --trasf False

# Analisi di trasferibilità (attacco generato su RawNet, valutato su AASIST)
python adversarial_evaluator_AllMetrics.py \
    --model_type full \
    --iter 10 \
    --eps 0.01 \
    --trasf True
```

**Produce:** Nella cartella `evaluation_results_adversarial_robustness/` (o `evaluation_results_adversarial_Rawnet/` per la trasferibilità):
- `evaluation_adversarial_Iter<n>_Eps<e>.json` — file JSON con tutte le metriche.

---

### `esc-50-aasist-eval.py`

**A cosa serve:** Valuta come il modello classifica i file audio del dataset **ESC-50** (Environmental Sound Classification), che contiene solo suoni ambientali reali (non speech). Serve a verificare che il modello non classifichi erroneamente i suoni ambientali come *spoof*.

**Cosa fa:**
1. Carica il modello Wav2Vec2 + AASIST.
2. Carica ogni file `.wav` del dataset ESC-50.
3. Esegue l'inferenza e ottiene lo score di probabilità spoof.
4. Salva le predizioni.

**Come si avvia:**

```bash
python esc-50-aasist-eval.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full
```

**Produce:**
- File di testo con le predizioni per ogni file audio ESC-50.

---

### `generate_graph.py`

**A cosa serve:** Genera **grafici a barre orizzontali** riassuntivi dei risultati degli esperimenti di rumore ambientale, mostrando l'andamento delle metriche al variare del tipo di rumore e del livello SNR.

**Cosa fa:**
1. Legge tutti i file `.json` prodotti da `ambient_noise_evaluator_AllMetrics.py`.
2. Organizza i dati per metrica, tipo di rumore e livello SNR.
3. Genera un grafico per ogni metrica (EER, Recovery Rate, Attack Effect, Attack Success Rate).
4. Ogni grafico mostra i 4 tipi di rumore come gruppi di barre, per i 4 livelli SNR (10, 15, 25, 35 dB).

> **Nota:** I percorsi `base_path` e `model_type` sono definiti come variabili interne allo script; modificali direttamente prima dell'esecuzione.

**Come si avvia:**

```bash
python generate_graph.py
```

**Produce:** Nella cartella `graph/<model_type>/summary/`:
- `plot_<model_type>_eer.png`
- `plot_<model_type>_recovery_rate.png`
- `plot_<model_type>_attack_effect.png`
- `plot_<model_type>_attack_success_rate.png`

---

## 5. Pipeline Completa di Esecuzione

Di seguito l'**ordine corretto** per eseguire tutti gli step del progetto da zero:

```
Step 1: Estrazione features
        └─ extract_features_optimized_og.py  (per codecfake, asvspoof2019, asvspoof5)

Step 2: Addestramento modello
        └─ main_train.py

Step 3: Valutazione baseline
        ├─ evaluate_model_asvspoof5.py       (test set → produce sample_probability.txt)
        └─ evaluate_model_co-dataset.py      (dev set co-dataset)

Step 4a: Esperimento Rumore Ambientale
        ├─ ambient_noise_eval_script.bash    (genera predizioni con rumore per tutti i rumori/SNR)
        └─ ambient_noise_evaluator_AllMetrics.py  (calcola metriche per ogni combinazione)

Step 4b: Esperimento Avversariale
        ├─ adversarial_sample_generation.py  (genera campioni PGD e predizioni)
        └─ adversarial_evaluator_AllMetrics.py   (calcola metriche di robustezza)

Step 5: Analisi ESC-50
        └─ esc-50-aasist-eval.py

Step 6: Generazione Grafici
        └─ generate_graph.py
```

---

## 6. Output Prodotti da Ogni Script

| Script | Output principale | Cartella |
|--------|-------------------|----------|
| `extract_features_optimized_og.py` | File `.pt` delle features | `features/<dataset>/<model_type>/` |
| `main_train.py` | Modello addestrato `.pt` | `../Models/cotrain_AASIST_FULL/` |
| `evaluate_model_asvspoof5.py` | `metrics.txt`, `sample_probability.txt` | `evaluation_results/<model_type>/` |
| `evaluate_model_co-dataset.py` | `metrics.txt`, `sample_probability.txt` | `evaluation_results_codatasets/<model_type>/` |
| `ambient_noise_generator_new.py` | `sample_probability_ambient_adversarial_SNR<n>.txt` | `ambient_noise_probabilities/aasist_<model_type>/<rumore>/` |
| `ambient_noise_evaluator_AllMetrics.py` | `evaluation_ambiental_noisy_snr<n>.json` | `evaluation_results_ambient_noise/aasist_<model_type>/<rumore>/` |
| `adversarial_sample_generation.py` | `sample_probability_adversarial_*.txt`, `metrics_*.txt` | `adversarial_results/` |
| `adversarial_evaluator_AllMetrics.py` | `evaluation_adversarial_Iter<n>_Eps<e>.json` | `evaluation_results_adversarial_robustness/` |
| `generate_graph.py` | Grafici `.png` | `graph/<model_type>/summary/` |

---

## 7. Glossario delle Metriche

| Metrica | Significato |
|---------|-------------|
| **EER** (Equal Error Rate) | Il punto in cui il tasso di false rejection (FRR) eguaglia il tasso di false acceptance (FAR). Più basso = meglio. |
| **minDCF** | Costo minimo di detection pesato. Considera il costo asimmetrico di accettare uno spoof (Cfa=10) vs rifiutare un bonafide (Cmiss=1). |
| **AUC** | Area sotto la curva ROC. Più alta = meglio (1.0 = perfetto). |
| **Attack Success Rate** | % di campioni spoof, originariamente classificati correttamente, che dopo l'attacco vengono classificati come bonafide. |
| **Recovery Rate** | % di campioni spoof, originariamente classificati male, che dopo l'attacco vengono classificati correttamente (effetto involontariamente correttivo del rumore). |
| **Effetto Attacco** | Differenza netta: (misclassificazioni corrette − misclassificazioni inverse) / tot campioni. Misura l'impatto reale dell'attacco. |
| **FAR** | False Acceptance Rate — percentuale di spoof accettati come bonafide. |
| **FRR** | False Rejection Rate — percentuale di bonafide rifiutati come spoof. |
| **SNR** | Signal-to-Noise Ratio — rapporto tra la potenza del segnale originale e quella del disturbo aggiunto (in dB). Più alto = disturbo meno percettibile. |
| **TP / TN / FP / FN** | True Positive (spoof correttamente identificati), True Negative (bonafide correttamente identificati), False Positive (bonafide classificati come spoof), False Negative (spoof classificati come bonafide). |
