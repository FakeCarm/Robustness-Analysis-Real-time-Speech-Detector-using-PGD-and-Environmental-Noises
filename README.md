# Anti-Spoofing Audio Detection — Project Documentation

This project implements an **audio deepfake detection system (anti-spoofing)** based on neural models (AASIST + Wav2Vec2). The system is able to distinguish real audio (*bonafide*) from artificially generated audio (*spoof*). The project also includes an evaluation of the model's robustness against **adversarial attacks** (PGD) and **ambient noise**.

---

## 📬 Contact

> For any question or clarification **related to the code**, feel free to reach out to me directly.

---

## Table of Contents

1. [General Architecture](#1-general-architecture)
2. [Requirements and Dependencies](#2-requirements-and-dependencies)
3. [Expected Folder Structure](#3-expected-folder-structure)
4. [File Descriptions](#4-file-descriptions)
5. [Full Execution Pipeline](#5-full-execution-pipeline)
6. [Output Summary per Script](#6-output-summary-per-script)
7. [Metrics Glossary](#7-metrics-glossary)

---

## 1. General Architecture

The system is composed of two main components in cascade:

```
Raw audio (.flac/.wav)
        │
        ▼
[ Wav2Vec2-XLS-R-300M ]   ← Feature extractor (pre-trained model)
        │
        ▼
[ AASIST Classifier ]     ← Bonafide / spoof classifier
        │
        ▼
  Probability score       (0.0 = bonafide, 1.0 = spoof)
        │
        ▼
  Decision with threshold (default: 0.714 for "full" model, 0.7 for "light")
```

Two model variants are available:
- **`full`**: uses all 24 transformer layers of Wav2Vec2.
- **`light`**: uses only the first 5 transformer layers of Wav2Vec2 — faster but slightly less accurate.

---

## 2. Requirements and Dependencies

```bash
pip install torch torchaudio transformers librosa scikit-learn numpy tqdm matplotlib adversarial-robustness-toolbox
```

Main libraries:
- `torch`, `torchaudio` — deep learning and audio processing
- `transformers` — loading Wav2Vec2 from HuggingFace
- `librosa` — audio loading and manipulation
- `scikit-learn` — evaluation metrics (EER, AUC, confusion matrix)
- `art` (Adversarial Robustness Toolbox) — PGD adversarial attacks
- `matplotlib` — graph generation

---

## 3. Expected Folder Structure

```
project/
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
├── features/                     ← Pre-extracted features (.pt files)
│   ├── codecfake/full/
│   ├── asvspoof2019/full/
│   └── asvspoof5/full/
│
├── evaluation_results/           ← Test set evaluation results
├── evaluation_results_ambient_noise/
├── evaluation_results_adversarial_robustness/
├── ambient_noise_probabilities/
├── adversarial_results/
│
├── ../ASVspoof5/                 ← ASVspoof5 dataset (raw audio)
├── ../ASVspoof19/LA/             ← ASVspoof2019 dataset
├── ../Codecfake/                 ← Codecfake dataset
├── ../ESC-50/train/              ← ESC-50 dataset (ambient noise audio files)
└── ../Models/cotrain_AASIST_FULL/ ← Trained model checkpoint (.pt)
```

---

## 4. File Descriptions

---

### `raw_dataset.py`

**Purpose:** Defines PyTorch `Dataset` classes for loading **raw audio files** (`.flac` and `.wav`), without any feature pre-processing. It is mainly used by `extract_features_optimized_og.py` to load audio before extracting features with Wav2Vec2.

**What it does:** Contains the following Dataset classes:
- `ASVspoof2019Raw` — loads ASVspoof2019 audio files directly from disk.
- `ASVspoof5` — loads only the *spoof* samples from the ASVspoof5 dataset.
- `codecfake` — loads Codecfake audio files by reading the corresponding protocol file (a `.txt` file specifying filenames and labels).
- `codecfake_eval` — evaluation-only version of the Codecfake dataset.
- `ASVspoof2019LAeval` — evaluation-only version of the ASVspoof2019 LA dataset.

Each class:
1. Reads the protocol file (`.txt` or `.tsv`) to know which files to load and which labels to assign.
2. Verifies that each file exists on disk before adding it to the list.
3. Implements `__getitem__`, which loads the waveform at 16kHz in mono and returns the tuple `(waveform, filename, label)`.

**How to run:** Not run directly. It is a module imported by other scripts.

---

### `extract_features_optimized_og.py`

**Purpose:** Performs **audio feature pre-extraction** using the Wav2Vec2-XLS-R-300M model. This step transforms raw audio files into feature tensors (`.pt` files) saved to disk, which are then used by the trainer and evaluators without needing to re-run the extractor each time. This is a **mandatory step** that must be completed **before** training.

**What it does:**
1. Loads the Wav2Vec2 model from HuggingFace (`facebook/wav2vec2-xls-r-300m`).
2. For each audio file in the chosen dataset, loads it at 16kHz, normalizes it, and feeds it to the model.
3. Extracts the hidden states from the target layer and saves them as `.pt` files in the output folder.
4. If `--skip_existing True`, skips already processed files to resume an interrupted extraction.
5. Supports both `full` mode (all 24 layers) and `light` mode (only the first 5 layers).

**How to run:**

```bash
# Feature extraction for Codecfake (full model)
python extract_features_optimized_og.py \
    --dataset codecfake \
    --dataset_path ../Codecfake/upload_zenodo \
    --output_path features \
    --model_type full \
    --skip_existing False

# Feature extraction for ASVspoof2019
python extract_features_optimized_og.py \
    --dataset asvspoof2019 \
    --dataset_path ../ASVspoof19/LA \
    --output_path features \
    --model_type full \
    --skip_existing False

# Feature extraction for ASVspoof5
python extract_features_optimized_og.py \
    --dataset asvspoof5 \
    --dataset_path ../ASVspoof5 \
    --output_path features \
    --model_type full \
    --skip_existing False
```

**Produces:** `.pt` files in `features/<dataset>/<model_type>/<split>/xls/`, one per audio sample. The filename includes the original filename and the label (e.g., `LA_T_1000137_label_1.pt`).

---

### `dataset_features.py`

**Purpose:** Defines PyTorch `Dataset` classes for loading **pre-extracted features** (`.pt` files). It is used by `main_train.py`, `evaluate_model_asvspoof5.py`, and `evaluate_model_co-dataset.py` in place of `raw_dataset.py`, since the features are already ready on disk.

**What it does:** Contains three Dataset classes:
- `ASVspoof2019` — loads pre-extracted ASVspoof2019 features from `features/asvspoof2019/`.
- `ASVspoof5` — loads pre-extracted ASVspoof5 features from `features/asvspoof5/`.
- `codecfake` — loads pre-extracted Codecfake features from `features/codecfake/`.

Each class:
1. Lists all `.pt` files in the folder corresponding to dataset + split + feature type.
2. Extracts the label from the filename (the last field before `.pt`, separated by `_`).
3. Implements `__getitem__`, which loads the feature tensor with `torch.load` and returns `(featureTensor, filename, label)`.

**How to run:** Not run directly. It is a module imported by other scripts.

---

### `main_train.py`

**Purpose:** Main script for **training** the AASIST classifier. Trains the model to distinguish bonafide audio from spoof audio using the pre-extracted Wav2Vec2 features.

**What it does:**
1. Reads parameters from the command line (dataset, epochs, batch size, learning rate, etc.).
2. Loads training and validation datasets from the `features/` folder using `dataset_features.py`.
3. If `--train_task co-train`, trains the model on **both** Codecfake and ASVspoof2019 datasets simultaneously (co-training).
4. If `--CSAM True`, applies the CSAM (Class-Specific Attention Module) technique during training.
5. If `--use_weighted_sampler True`, uses a weighted sampler to balance the bonafide/spoof classes.
6. At each epoch, computes validation metrics (EER, minDCF) and saves the best model checkpoint.
7. Applies a weighted loss to handle class imbalance between bonafide and spoof samples.

**How to run:**

```bash
# Co-training on Codecfake + ASVspoof2019
python main_train.py \
    --path_to_features features/codecfake/full \
    --path_to_features1 features/asvspoof2019/full \
    --out_fold ../Models/cotrain_AASIST_FULL \
    --train_task co-train \
    --CSAM True \
    --feat xls \
    --use_weighted_sampler True
```

**Produces:** In the `--out_fold` folder:
- `anti-spoofing_feat_model.pt` — best model checkpoint (used by all other scripts).
- `training_log.txt` — log file with metrics for each epoch.

---

### `evaluate_model_asvspoof5.py`

**Purpose:** Evaluates the **baseline performance** of the trained model on the **ASVspoof5** test set, using pre-extracted features. This is the starting point before any robustness analysis.

**What it does:**
1. Loads the saved model (`.pt`) and the ASVspoof5 dataset from the `features/` folder.
2. Runs inference on all test set samples.
3. Computes metrics: EER, minDCF, AUC, Accuracy, Precision, Recall, F1, Specificity, and latency.
4. Saves the per-sample predictions to a text file for subsequent analyses.

**How to run:**

```bash
python evaluate_model_asvspoof5.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --data features/asvspoof5 \
    --output_dir evaluation_results \
    --model_type full
```

**Produces:** In `evaluation_results/<model_type>/`:
- `metrics.txt` — all evaluation and latency metrics.
- `sample_probability.txt` — one line per sample: `<filename> <score> <true_label> <prediction>`. **This file is critical:** it is used as input by `ambient_noise_evaluator_AllMetrics.py` and `adversarial_evaluator_AllMetrics.py`.

---

### `evaluate_model_co-dataset.py`

**Purpose:** Evaluates the **model performance on the dev set** of the co-training datasets (ASVspoof2019 and/or Codecfake), in order to measure generalization on data not seen during testing.

**What it does:** Identical to `evaluate_model_asvspoof5.py`, but loads ASVspoof2019 (or Codecfake) from the features folder and evaluates on the `dev` split.

**How to run:**

```bash
python evaluate_model_co-dataset.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --dataset_asv features/asvspoof2019 \
    --dataset_codec features/codecfake \
    --output_dir evaluation_results_codatasets \
    --model_type full
```

**Produces:** In `evaluation_results_codatasets/<model_type>/`:
- `metrics.txt` — evaluation metrics on the dev set.
- `sample_probability.txt` — per-sample predictions on the dev set.

---

### `taratura_dev_set.py`

**Purpose:** **Calibration** script that evaluates the model on the combined dev set of Codecfake and ASVspoof2019. The goal is to find the optimal decision threshold (EER threshold) to be used later on the test set.

**What it does:** Works almost identically to `evaluate_model_co-dataset.py`, but:
1. Merges the two validation datasets into a single `ConcatDataset`.
2. Computes the optimal EER threshold on the combined dataset.
3. Saves the per-sample probabilities.

> ⚠️ **Note:** The file contains a minor bug (`if __main__ == "__main__"` instead of `if __name__ == "__main__"`) that must be fixed before running.

**How to run:**

```bash
python taratura_dev_set.py \
    --model_path ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full \
    --path_to_features features/codecfake/full \
    --path_to_features1 features/asvspoof2019/full \
    --out_fold ./taratura_results
```

**Produces:** In `taratura_results/<model_type>/`:
- `metrics.txt` — metrics and EER threshold computed on the dev set.
- `sample_probability.txt` — per-sample predictions.

---

### `eval_metrics.py`

**Purpose:** Utility module for computing **anti-spoofing evaluation metrics**. Not meant to be run directly — it is imported by other scripts.

**What it does:** Contains the following functions:
- `compute_det_curve(target_scores, nontarget_scores)` — computes the DET curve (False Rejection Rate vs False Acceptance Rate) across thresholds.
- `compute_eer(target_scores, nontarget_scores)` — computes the **Equal Error Rate** (EER), i.e., the point where FRR = FAR. Also returns the frr, far vectors and corresponding thresholds.
- `compute_tDCF(...)` — computes the **tandem Detection Cost Function** (t-DCF), the metric used in ASVspoof 2019.
- `obtain_asv_error_rates(...)` — computes error rates for the ASV (Automatic Speaker Verification) system.

**How to run:** Not run directly. Imported with `import eval_metrics as em`.

---

### `ambient_noise_generator_new.py`

**Purpose:** Simulates a **realistic ambient noise attack** by mixing ESC-50 audio clips (wind, rain, crowd, children) with ASVspoof5 audio samples at a controlled Signal-to-Noise Ratio (SNR). It then evaluates how the added noise affects the model's predictions.

**What it does:**
1. Loads the Wav2Vec2 + AASIST model through the inference pipeline.
2. Loads a noise audio file from the ESC-50 dataset (e.g., `rain.wav`).
3. For each *spoof* sample in the ASVspoof5 test set, mixes the noise with the original audio at the specified target SNR.
4. Runs model inference on the noisy audio.
5. Saves the per-sample predictions (score and predicted label) to a text file.

**How to run:**

```bash
# Single run
python ambient_noise_generator_new.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full \
    --rumore pioggia \
    --snr 10

# Background run with log
nohup python ambient_noise_generator_new.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full \
    --rumore pioggia \
    --snr 10 > pio10.txt &
```

Valid values for `--rumore`: `pioggia` (rain), `vento` (wind), `folla` (crowd), `bambini` (children) — these must correspond to `.wav` files in `../ESC-50/train/`.

**Produces:** In `ambient_noise_probabilities/aasist_<model_type>/<rumore>/`:
- `sample_probability_ambient_adversarial_SNR<snr>.txt` — one line per sample: `<filename> <score> <prediction>`.

---

### `ambient_noise_eval_script.bash`

**Purpose:** Bash script for **sequential automation** of `ambient_noise_generator_new.py`, running it across all combinations of noise type and SNR level, without parallelism (to avoid GPU resource conflicts).

**What it does:**
1. Defines arrays of noise types (`bambini`, `folla`) and SNR levels (`35`, `25`, `15`, `10`).
2. Runs `ambient_noise_generator_new.py` sequentially for each combination.
3. Saves the log of each run to a separate file (e.g., `bam35.txt`, `fol10.txt`).

**How to run:**

```bash
# Make executable (first time only)
chmod +x ambient_noise_eval_script.bash

# Run in background with log
nohup ./ambient_noise_eval_script.bash \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full > ambfull.txt &
```

> **Note:** The `--model` and `--model_type` values are set as internal variables (`MODEL`, `MODEL_TYPE`) directly inside the script. Edit them there before running.

**Produces:** All `sample_probability_ambient_adversarial_SNR<snr>.txt` files for every noise/SNR combination.

---

### `ambient_noise_evaluator_AllMetrics.py`

**Purpose:** Computes all **ambient noise robustness metrics** by comparing the model's original predictions (without noise) to those obtained with noise added.

**What it does:**
1. Reads baseline predictions from `evaluation_results/<model_type>/sample_probability.txt`.
2. Reads noisy predictions from `ambient_noise_probabilities/aasist_<model_type>/<rumore>/sample_probability_ambient_adversarial_SNR<snr>.txt`.
3. For each *spoof* sample: replaces the original score with the one obtained with noise.
4. Computes the following metrics:
   - **EER** (Equal Error Rate)
   - **minDCF** (minimum Detection Cost Function)
   - **AUC** (Area Under the ROC Curve)
   - **Accuracy**, **Recall**, **Precision**
   - **Attack Success Rate** — percentage of spoof samples that were originally correctly classified but are now classified as bonafide after adding noise.
   - **Recovery Rate** — percentage of spoof samples that were originally misclassified but are now correctly classified (unintentional corrective effect of noise).
   - **Attack Effect** — net difference: (correct misclassifications − inverse misclassifications) / total samples.
   - **TP, TN, FP, FN** — confusion matrix values.

**How to run:**

```bash
python ambient_noise_evaluator_AllMetrics.py \
    --snr 15 \
    --rumore pioggia \
    --model_type full
```

**Produces:** In `evaluation_results_ambient_noise/aasist_<model_type>/<rumore>/`:
- `evaluation_ambiental_noisy_snr<snr>.json` — JSON file with all metrics for that noise/SNR combination.

---

### `adversarial_sample_generation.py`

**Purpose:** Generates **adversarial audio samples** using the **PGD (Projected Gradient Descent)** attack through the ART library (Adversarial Robustness Toolbox). The goal is to modify spoof audio as minimally as possible (imperceptible perturbation) so that the model incorrectly classifies it as bonafide.

**What it does:**
1. Loads the full model (Wav2Vec2 + AASIST) as a differentiable pipeline.
2. Wraps the pipeline in an ART `PyTorchClassifier`.
3. For each spoof sample in the test set, applies the PGD attack with different combinations of `max_iter` (number of iterations) and `eps_step` (perturbation step size).
4. Runs model inference on the perturbed audio.
5. Computes metrics such as post-attack accuracy, attack success rate, and perturbation SNR.
6. Saves predictions and metrics.

**How to run:**

> ⚠️ In this script the argument parser is commented out; the model and dataset paths are defined directly in the last lines of `__main__`. Edit the variables `model_path` and `dataset_path` before running.

```bash
python adversarial_sample_generation.py
```

**Produces:** In `adversarial_results/`:
- `sample_probability_adversarial_MaxIters<n>_EpsStep<e>.txt` — per-sample predictions for each parameter combination.
- `metrics_<n>_<e>.txt` — performance metrics for each combination.

---

### `adversarial_evaluator_AllMetrics.py`

**Purpose:** Computes all **adversarial robustness metrics** by comparing baseline predictions to those obtained on PGD-perturbed samples. Works analogously to `ambient_noise_evaluator_AllMetrics.py` but for adversarial attacks.

**What it does:**
1. Reads baseline predictions from `evaluation_results/sample_probability.txt`.
2. Reads adversarial predictions from `adversarial_results/sample_probability_adversarial_MaxIter<n>_EpsStep<e>.txt`.
3. Computes the same metrics as `ambient_noise_evaluator_AllMetrics.py` (EER, minDCF, AUC, Attack Success Rate, Recovery Rate, etc.).
4. Supports **transferability mode** (`--trasf True`): evaluates whether adversarial examples generated on RawNet can fool AASIST (black-box transfer attack).

**How to run:**

```bash
# Robustness analysis (attack on the same model)
python adversarial_evaluator_AllMetrics.py \
    --model_type full \
    --iter 10 \
    --eps 0.01 \
    --trasf False

# Transferability analysis (attack generated on RawNet, evaluated on AASIST)
python adversarial_evaluator_AllMetrics.py \
    --model_type full \
    --iter 10 \
    --eps 0.01 \
    --trasf True
```

**Produces:** In `evaluation_results_adversarial_robustness/` (or `evaluation_results_adversarial_Rawnet/` for transferability):
- `evaluation_adversarial_Iter<n>_Eps<e>.json` — JSON file with all metrics.

---

### `esc-50-aasist-eval.py`

**Purpose:** Evaluates how the model classifies audio files from the **ESC-50** dataset (Environmental Sound Classification), which contains only real ambient sounds (non-speech). Used to verify that the model does not incorrectly classify ambient sounds as *spoof*.

**What it does:**
1. Loads the Wav2Vec2 + AASIST model.
2. Loads each `.wav` file from the ESC-50 dataset.
3. Runs inference and obtains the spoof probability score.
4. Saves the per-sample predictions.

**How to run:**

```bash
python esc-50-aasist-eval.py \
    --model ../Models/cotrain_AASIST_FULL/anti-spoofing_feat_model.pt \
    --model_type full
```

**Produces:**
- A text file with predictions for each ESC-50 audio file.

---

### `generate_graph.py`

**Purpose:** Generates **horizontal bar charts** summarizing the results of the ambient noise experiments, showing how metrics vary across noise type and SNR level.

**What it does:**
1. Reads all `.json` files produced by `ambient_noise_evaluator_AllMetrics.py`.
2. Organizes the data by metric, noise type, and SNR level.
3. Generates one chart per metric (EER, Recovery Rate, Attack Effect, Attack Success Rate).
4. Each chart shows the 4 noise types as groups of bars, across the 4 SNR levels (10, 15, 25, 35 dB).

> **Note:** The `base_path` and `model_type` variables are defined directly inside the script; edit them before running.

**How to run:**

```bash
python generate_graph.py
```

**Produces:** In `graph/<model_type>/summary/`:
- `plot_<model_type>_eer.png`
- `plot_<model_type>_recovery_rate.png`
- `plot_<model_type>_attack_effect.png`
- `plot_<model_type>_attack_success_rate.png`

---

## 5. Full Execution Pipeline

Below is the **correct order** to run all project steps from scratch:

```
Step 1: Feature extraction
        └─ extract_features_optimized_og.py  (for codecfake, asvspoof2019, asvspoof5)

Step 2: Model training
        └─ main_train.py

Step 3: Baseline evaluation
        ├─ evaluate_model_asvspoof5.py       (test set → produces sample_probability.txt)
        └─ evaluate_model_co-dataset.py      (co-dataset dev set)

Step 4a: Ambient Noise Experiment
        ├─ ambient_noise_eval_script.bash    (generates noisy predictions for all noise/SNR combos)
        └─ ambient_noise_evaluator_AllMetrics.py  (computes metrics for each combination)

Step 4b: Adversarial Attack Experiment
        ├─ adversarial_sample_generation.py  (generates PGD samples and predictions)
        └─ adversarial_evaluator_AllMetrics.py   (computes robustness metrics)

Step 5: ESC-50 Analysis
        └─ esc-50-aasist-eval.py

Step 6: Graph Generation
        └─ generate_graph.py
```

---

## 6. Output Summary per Script

| Script | Main Output | Folder |
|--------|-------------|--------|
| `extract_features_optimized_og.py` | Feature `.pt` files | `features/<dataset>/<model_type>/` |
| `main_train.py` | Trained model `.pt` | `../Models/cotrain_AASIST_FULL/` |
| `evaluate_model_asvspoof5.py` | `metrics.txt`, `sample_probability.txt` | `evaluation_results/<model_type>/` |
| `evaluate_model_co-dataset.py` | `metrics.txt`, `sample_probability.txt` | `evaluation_results_codatasets/<model_type>/` |
| `ambient_noise_generator_new.py` | `sample_probability_ambient_adversarial_SNR<n>.txt` | `ambient_noise_probabilities/aasist_<model_type>/<noise>/` |
| `ambient_noise_evaluator_AllMetrics.py` | `evaluation_ambiental_noisy_snr<n>.json` | `evaluation_results_ambient_noise/aasist_<model_type>/<noise>/` |
| `adversarial_sample_generation.py` | `sample_probability_adversarial_*.txt`, `metrics_*.txt` | `adversarial_results/` |
| `adversarial_evaluator_AllMetrics.py` | `evaluation_adversarial_Iter<n>_Eps<e>.json` | `evaluation_results_adversarial_robustness/` |
| `generate_graph.py` | `.png` charts | `graph/<model_type>/summary/` |

---

## 7. Metrics Glossary

| Metric | Meaning |
|--------|---------|
| **EER** (Equal Error Rate) | The point at which the False Rejection Rate (FRR) equals the False Acceptance Rate (FAR). Lower is better. |
| **minDCF** | Minimum weighted detection cost. Accounts for the asymmetric cost of accepting a spoof (Cfa=10) vs. rejecting a bonafide (Cmiss=1). |
| **AUC** | Area Under the ROC Curve. Higher is better (1.0 = perfect). |
| **Attack Success Rate** | % of spoof samples that were originally correctly classified but are now classified as bonafide after the attack. |
| **Recovery Rate** | % of spoof samples that were originally misclassified but are now correctly classified (unintentional corrective effect of noise/perturbation). |
| **Attack Effect** | Net impact: (correct misclassifications − inverse misclassifications) / total samples. Measures the real-world impact of the attack. |
| **FAR** | False Acceptance Rate — percentage of spoof samples accepted as bonafide. |
| **FRR** | False Rejection Rate — percentage of bonafide samples rejected as spoof. |
| **SNR** | Signal-to-Noise Ratio — ratio between the power of the original signal and the added disturbance (in dB). Higher = less perceptible disturbance. |
| **TP / TN / FP / FN** | True Positive (spoof correctly identified), True Negative (bonafide correctly identified), False Positive (bonafide classified as spoof), False Negative (spoof classified as bonafide). |
