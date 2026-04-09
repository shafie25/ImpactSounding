# JapanesePaper2Code — Acoustic Impact Hammer Testing Pipeline

This project re-implements and extends the pipeline from:

> **"Artificial intelligence enhanced automatic identification for concrete cracks using acoustic impact hammer testing"**
> Alhebrawi, Huang, Wu — Ibaraki University, Japan & Southeast University, China
> *Journal of Civil Structural Health Monitoring* (2023) 13:469–484
> DOI: 10.1007/s13349-022-00651-8

The paper proposes an automated AI pipeline to detect concrete cracks and classify their width and depth from acoustic impact hammer recordings — removing the need for skilled human inspectors.

---

## About the Dataset

**Specimens:** 6 reinforced concrete blocks (400 × 100 × 100 mm), compressive strength ~29 MPa. All cracks are artificial, created by embedding greased polyethylene sheets at 45° before casting and removing them after curing.

| Specimen | Crack Width | Crack Length | Crack Depth (vertical) |
|----------|-------------|--------------|------------------------|
| 1.1 | 0.2 mm | 80 mm | 40 mm deep |
| 1.2 | 0.4 mm | 80 mm | 40 mm deep |
| 1.3 | 0.6 mm | 80 mm | 40 mm deep |
| 2.1 | 0.2 mm | 40 mm | 20 mm deep |
| 2.2 | 0.4 mm | 40 mm | 20 mm deep |
| 2.3 | 0.6 mm | 40 mm | 20 mm deep |

**Crack geometry:** Cracks are diagonal at 45°. The depths listed in Table 1 of the paper (56.57 mm for Group 1, 28.28 mm for Group 2) are the actual diagonal crack path lengths. The vertical penetration depths — what the paper calls "40 mm deep" and "20 mm deep" — are the horizontal/vertical projections (56.57 / √2 ≈ 40 mm, 28.28 / √2 ≈ 20 mm).

**Crack fabrication:** Each crack was created by fixing a polyethylene sheet inside the mold before casting and pulling it out after curing. Crack width was controlled by layering: 1 layer = 0.2 mm, 2 layers = 0.4 mm, 3 layers = 0.6 mm. Both sides of the sheet were greased with heavy oil grease to ease removal.

**Hammering positions:** Lines were drawn at 2 cm intervals perpendicular to the crack. `C02`, `C04`, `C06`... = positions on the **cracked side** at 2 cm, 4 cm, 6 cm... from the crack. `N02`, `N04`... = same distances on the **intact side**. Our `Class_sounds/` folder only contains the two closest cracked positions (C02, C04) and intact specimens.

**Positions used per task (paper's Table 2):**

| Task | Class | Positions used |
|------|-------|----------------|
| Crack Detection | Intact | Series 1: C06, C08, C10 — Series 2: C04, C06, C08, C10 — all N positions |
| Crack Detection | Cracked | Series 1: C02, C04 — Series 2: C02 only |
| Crack Width | 0.2 / 0.4 / 0.6 mm | Series 1 (C02+C04) and Series 2 (C02 only) per width |
| Crack Depth | 20 mm | Series 2 specimens (C02 only) |
| Crack Depth | 40 mm | Series 1 specimens (C04 only) |

**Recording setup:** Sony ECM-PCV80U condenser microphone inside a foam-lined cardboard box (500 × 400 × 40 mm high-density foam) to isolate ambient noise. 200 hammer hits recorded per position per specimen.

**Class balancing (ESS):** The paper used Equal Size Sampling (ESS) — downsampling the majority class to match the minority — before training. After ESS: crack detection = 3,600 total (1,800 cracked + 1,800 intact), 70/30 split → 2,520 train / 1,080 test. Crack width = 1,800 total, 1,260 train / 540 test. Crack depth = 1,200 total, 840 train / 360 test.

**Tuned SVM hyperparameters (RBF kernel, Bayesian optimization, 100 iterations):**

| Task | C | γ |
|------|---|---|
| Crack detection | 5.33 | 0.37 |
| Crack width | 7.04 | 0.80 |
| Crack depth | 7.53 | 0.91 |

---

## Paper's Approach (3-Level Hierarchy)

The paper uses a **hierarchical two-level detection structure**:

```
Level 1 — Crack Detection (Cracked vs. Intact)
        |
        +--> if Cracked:
                |
                +--> Level 2a — Crack Width  (0.2 mm / 0.4 mm / 0.6 mm)
                +--> Level 2b — Crack Depth  (20 mm / 40 mm)
```

Width and depth are **separate independent classifiers**, not a single multi-output model.

**Three feature approaches compared in the paper:**
1. **Dominant Frequency (Df)** — FFT peak frequency. Can detect crack presence but cannot reliably distinguish width or depth.
2. **Frequency Feature Value (Vf)** — amplitude-weighted frequency spread. Better for shallow (20 mm) cracks, fails for deep (40 mm) cracks.
3. **MFCC** — best performing. 12 coefficients (coefficient #1 dropped as too noisy), **max and mean** extracted per coefficient = **24 features total**. Chunk size: 512 samples, 25% overlap. Implemented in KNIME with JAudio plugin.

**Three ML models compared:** Fuzzy Rules (FR), Gradient Boosted Trees (GBT), SVM — SVM won all tasks.

**SVM setup:** RBF kernel, C and γ tuned via Bayesian optimization (TPE, 100 iterations), 5-fold CV for hyperparameter selection, **70/30 train/test split**, **Equal Size Sampling (ESS)** to handle class imbalance (downsample majority class).

### Paper's Results (SVM)

| Task | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Crack Detection (Cracked vs. Intact) | 97.59% | 97.95% | 97.22% | 97.58% |
| Crack Width (0.2 / 0.4 / 0.6 mm) | 99.44% | ~99.4% | ~99.4% | ~99.4% |
| Crack Depth (20 mm / 40 mm) | 96.67% | ~96.7% | ~96.7% | ~96.7% |

---

## Our Pipeline

```
Raw recordings (ALL data/*.wav)
        |
  [Hammer Segmentation]          <- Original Paper Files/
        |
Specimens/Split_sounds/
        |
  [Labeled by condition]
        |
Specimens/Class_sounds/
  Cracked/  |  Intact/           <- 10,036 labeled WAV files
        |
  [Labeled Feature Extraction]   <- our script
        |
  Labeled_Features.xlsx          <- 10,036 rows x 33 columns
  (Df, Vf, 26 MFCC stats + label)
        |
  [Classifiers]                  <- our scripts
        |
  Classifier Results/
  RandomForest | XGBoost | SVM | CNN_MFCC | CNN_1D | LSTM | MLP
```

---

## File Structure

```
JapanesePaper2Code/
├── Classifier Results/
│   ├── CrackDetection/          Level 1 — Cracked vs Intact
│   │   ├── RandomForest/        metrics.json, confusion_matrix.png, feature_importances.png
│   │   ├── XGBoost/             metrics.json, confusion_matrix.png, feature_importances.png
│   │   ├── SVM/                 metrics.json, confusion_matrix.png
│   │   ├── CNN_MFCC/            best_head.pt, metrics.json, confusion_matrix.png, training_curves.png
│   │   ├── CNN_1D/              best_model.pt, metrics.json, confusion_matrix.png, training_curves.png
│   │   ├── LSTM/                best_model.pt, metrics.json, confusion_matrix.png, training_curves.png
│   │   ├── MLP/                 best_model.pt, metrics.json, confusion_matrix.png, training_curves.png
│   │   ├── Comparison/          side-by-side bar charts across all classifiers
│   │   └── Visualization/       torchinfo summaries, ONNX exports, Grad-CAM heatmaps
│   ├── CrackWidth/              Level 2a — 0.2 / 0.4 / 0.6 mm
│   │   ├── RandomForest/        metrics.json, confusion_matrix.png, cv_scores.png
│   │   ├── XGBoost/             metrics.json, confusion_matrix.png, cv_scores.png
│   │   ├── SVM/                 metrics.json, confusion_matrix.png, cv_scores.png
│   │   ├── MLP/                 best_model.pt, metrics.json, confusion_matrix.png, training_curves.png
│   │   ├── CNN_1D/              best_model.pt, metrics.json, confusion_matrix.png, training_curves.png
│   │   └── Visualization/       torchinfo summaries, ONNX exports, Grad-CAM heatmaps (CNN_1D + MLP)
│   └── CrackDepth/              Level 2b — 20 mm / 40 mm (planned)
├── MFCC_Images/
│   ├── Cracked/                 1,800 × 224×224px MFCC spectrogram images
│   └── Intact/                  8,236 × 224×224px MFCC spectrogram images
├── Original Paper Files/        paper's preprocessing scripts (not used by our pipeline)
├── Sounds/                      200 demo WAVs provided with the paper
├── Specimens/
│   ├── Class_sounds/
│   │   ├── Cracked/             1,800 labeled WAV files (9 subfolders by specimen + position)
│   │   └── Intact/              8,236 labeled WAV files (6 subfolders by specimen)
│   └── Split_sounds/            impacts organized per specimen and position
├── Classifier_1DCNN.py          Level 1: 1D CNN on raw waveforms (crack detection)
├── Classifier_CNN_MFCC.py       Level 1: EfficientNet-B0 on MFCC spectrogram images
├── Classifier_Comparison.py     loads all metrics.json files and plots comparisons
├── Classifier_LSTM.py           Level 1: LSTM on MFCC frame sequences
├── Classifier_MLP.py            Level 1: MLP on tabular features
├── Classifier_RandomForest.py   Level 1: Random Forest on tabular features
├── Classifier_SVM.py            Level 1: SVM on tabular features
├── Classifier_XGBoost.py        Level 1: XGBoost on tabular features
├── Classifier_Width_Tabular.py  Level 2a: RF + XGBoost + SVM for crack width (specimen-level split)
├── Classifier_Width_MLP.py      Level 2a: MLP for crack width (specimen-level split)
├── Classifier_Width_1DCNN.py    Level 2a: 1D CNN for crack width (specimen-level split)
├── Labeled Feature Extraction.py  extracts Df, Vf, MFCC stats for all 10,036 WAVs
├── Labeled_Features.xlsx        10,036 rows × 33 columns feature table
├── MFCC_Image_Generator.py      converts each WAV to a 224×224 MFCC image
├── visualize_models.py          Level 1: torchinfo summaries, ONNX exports, Grad-CAM heatmaps
├── visualize_width_models.py    Level 2a: torchinfo summaries, ONNX exports, Grad-CAM heatmaps
├── JapanesePaper.pdf
├── README.md
└── TECHNICAL_NOTES.md
```

---

## Folders

### `Original Paper Files/`
Scripts and data files from the original research paper — not modified. These handle raw audio preprocessing and single-feature extraction on the 200 demo WAV files in `Sounds/`. **Not used by any of our scripts.**

| File | Purpose |
|------|---------|
| `Hammer Impact Detection & Segmentation.py` | Splits long recordings into individual impacts using silence detection, trims to 0.20s, filters noise |
| `MFCC Feature Extraction & Visualization.py` | Computes 13 MFCC coefficients per impact, saves as 194×195px JPG images (CNN-ready) |
| `Dominant Frequency (Df) Feature Extractor for Audio.py` | FFT-based peak frequency extraction, saves to Excel |
| `Frequency Feature Value (Vf) Extraction Procedure.py` | Energy-weighted frequency spread metric (Vf formula from the paper), saves to Excel |
| `Mel-Frequency Cepstral Coefficients (MFCC) feature extraction Knime workflow.knwf` | KNIME visual workflow version of the MFCC pipeline |
| `Dominant_Frequency_Df_Features.xlsx` | Df extraction output on the 200 demo files — superseded by `Labeled_Features.xlsx` |
| `FFT_Feature_Value_Vf.xlsx` | Vf extraction output on the 200 demo files — superseded by `Labeled_Features.xlsx` |

### `Sounds/`
200 demo WAV files (`000.wav` – `199.wav`) provided with the paper for demonstration only. Not used for ML.

### `Specimens/`
- `Split_sounds/` — impacts organized per specimen and position, output of the segmentation script
- `Class_sounds/` — **the real labeled dataset (10,036 WAV files)**
  - `Cracked/` — 9 subfolders by specimen + crack position (C02 / C04): 1,800 files total
  - `Intact/` — 6 subfolders by specimen: 8,236 files total

### `MFCC_Images/`
224×224px MFCC spectrogram images generated by `MFCC_Image_Generator.py`. Flat structure compatible with `torchvision.datasets.ImageFolder`.
- `Cracked/` — 1,800 images
- `Intact/` — 8,236 images

### `Classifier Results/`
Output folder for all ML results, organized by model and comparison. Each subfolder contains a `metrics.json` (loaded by `Classifier_Comparison.py`), confusion matrix, and training curves where applicable. The `Visualization/` subfolder is generated by `visualize_models.py`.

---

## Our Scripts

### `Labeled Feature Extraction.py`
Walks `Specimens/Class_sounds/`, extracts features from all 10,036 WAVs, and saves a single labeled table to `Labeled_Features.xlsx`.


**Features extracted per file (29 total):**
- `Df_Hz`, `Df_Amplitude` — dominant FFT frequency and its magnitude
- `Vf` — energy-weighted frequency spread
- `MFCC_1_mean` through `MFCC_13_mean` — mean of each MFCC coefficient
- `MFCC_1_std` through `MFCC_13_std` — std of each MFCC coefficient

**Differences from the paper's MFCC approach:** We use all 13 coefficients (paper drops #1), and extract mean + std (paper extracts max + mean). Both are valid — ours captures spread, theirs captures peaks.

**Metadata columns:** `Signal_ID`, `Label` (Cracked/Intact), `Crack_Position` (C02/C04/N/A), `Specimen`

---

### `Classifier_RandomForest.py`
Trains a Random Forest (300 trees, `class_weight=balanced`) on `Labeled_Features.xlsx`. Saves confusion matrix, feature importances, and CV score plots to `Classifier Results/RandomForest/`.

### `Classifier_XGBoost.py`
Trains an XGBoost classifier (`scale_pos_weight=4.58` for imbalance). Saves confusion matrix, feature importances, and CV score plots to `Classifier Results/XGBoost/`.

### `Classifier_SVM.py`
Trains an RBF-kernel SVM inside a StandardScaler pipeline (`class_weight=balanced`). Saves confusion matrix and CV score plots to `Classifier Results/SVM/`.

### `MFCC_Image_Generator.py`
Walks `Specimens/Class_sounds/` and generates a 224×224px MFCC spectrogram image for every WAV file. Saves to `MFCC_Images/Cracked/` and `MFCC_Images/Intact/` — a flat structure ready for `torchvision.datasets.ImageFolder`.

Uses identical parameters to `Labeled Feature Extraction.py` (`n_mfcc=13`, `sr=None`, default librosa hop/fft) so the images are consistent with the extracted features. The existing `Plots/` subfolders inside each specimen folder contain **waveform plots** (not spectrograms) generated by the segmentation script — those are for visual inspection only and are not used for ML.

**Output:** 1,800 Cracked + 8,236 Intact = 10,036 images.

---

### `Classifier_CNN_MFCC.py`
Trains an **EfficientNet-B0** CNN on the MFCC spectrogram images in `MFCC_Images/`. Uses a transfer learning / feature extraction approach optimized for CPU:

1. Runs all 10,036 images through the frozen pretrained backbone once (~4 min on CPU) to extract 1,280-dimensional feature vectors.
2. Trains a small MLP head (`Linear(1280→256) → ReLU → Dropout → Linear(256→2)`) on those cached features for 50 epochs.

**Split:** 80/10/10 stratified train/val/test. **Class weighting:** `CrossEntropyLoss` with `weight=[2.79, 0.61]` for Cracked/Intact imbalance. Best head weights saved to `Classifier Results/CNN_MFCC/best_head.pt`.

---

### `Classifier_1DCNN.py`
Trains a **1D CNN** directly on raw audio waveforms — no feature engineering. Each WAV is loaded as a 4,410-sample float array and fed into a 4-layer Conv1d network trained from scratch.

- **Architecture:** Conv1d(1→32, k=64, s=4) → Conv1d(32→64, k=32, s=2) → Conv1d(64→128, k=16, s=2) → Conv1d(128→256, k=8, s=2) → GlobalAvgPool → Linear(256→64) → Dropout(0.3) → Linear(64→2). ~478K parameters.
- **Augmentation:** Gaussian noise (0.5% signal std) + random circular time shift (±5%) applied during training only.
- **Split:** 80/10/10 stratified train/val/test. **Class weighting:** same as CNN_MFCC (Cracked=2.788, Intact=0.609). Best weights saved to `Classifier Results/CNN_1D/best_model.pt`.

---

### `Classifier_LSTM.py`
Trains a **2-layer LSTM** on MFCC frame sequences. Each WAV is converted to a sequence of 9 MFCC frames (T=9, 13 coefficients per frame) using `hop_length=512` — the default librosa hop on 4,410 samples gives exactly 9 frames. Unlike the tabular classifiers which collapse MFCCs to mean/std statistics, the LSTM processes each frame in order and captures how spectral content evolves from impact through resonance to decay.

- **Architecture:** `LSTM(13→64, 2 layers, dropout=0.3)` → last hidden state → `Linear(64→32) → ReLU → Dropout → Linear(32→2)`. ~55K parameters.
- **Augmentation:** Gaussian noise on MFCC values + random time masking (1 frame zeroed per sample, training only).
- **Split:** 80/10/10 stratified. **Class weighting:** same as 1D CNN (Cracked=2.788, Intact=0.609). Best weights saved to `Classifier Results/LSTM/best_model.pt`.

---

### `Classifier_MLP.py`
Trains a **2-hidden-layer MLP** on the same 29 tabular features as SVM/XGBoost/RF. This is the fairest NN vs. classical comparison: identical features, different learner. Features are StandardScaler-normalized before training (MLPs are scale-sensitive; tree models are not).

- **Architecture:** `Linear(29→128) → BN → ReLU → Dropout(0.3) → Linear(128→64) → BN → ReLU → Dropout(0.3) → Linear(64→2)`. ~12K parameters.
- **Split:** 80/10/10 stratified. **Class weighting:** same as other NNs (Cracked=2.788, Intact=0.609). Best weights saved to `Classifier Results/MLP/best_model.pt`.

---

### `Classifier_Comparison.py`
Loads `metrics.json` from each classifier's output folder and produces side-by-side comparison plots in `Classifier Results/Comparison/`. **Run after all seven classifiers.**

---

### `visualize_models.py`
Loads the trained `.pt` model weights and produces three types of visualization in `Classifier Results/Visualization/`:

1. **Architecture summaries** (`cnn1d_summary.txt`, `cnn_mfcc_head_summary.txt`, `lstm_summary.txt`, `mlp_summary.txt`) — layer-by-layer breakdown showing input/output shapes and parameter counts per layer, generated with `torchinfo`.

2. **ONNX exports** (`cnn1d.onnx`, `cnn_mfcc_head.onnx`, `lstm.onnx`, `mlp.onnx`) — drag any file into [netron.app](https://netron.app) for an interactive visual graph of the architecture.

3. **Grad-CAM heatmaps** (`gradcam_cracked_sample1-3.png`, `gradcam_intact_sample1-3.png`) — for the 1D CNN only. Each plot overlays a color heatmap on the raw waveform showing which milliseconds of the impact sound the model focused on when making its prediction. Red/warm = high importance, blue/cool = low importance. Three samples per class are generated to check consistency across different recordings.

**Dependencies:** `torchinfo`, `onnx`, `onnxscript` (install with `pip install torchinfo onnx onnxscript`).

---

## Our Results vs. Paper (Crack Detection)

| | Paper SVM | Our SVM | Our XGBoost | Our RF | Our MFCC CNN | Our 1D CNN | Our LSTM | Our MLP |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 97.59% | 99.70% | 99.25% | 98.51% | 92.33% | **99.75%** | 99.20% | 99.60% |
| Precision (Cracked) | 97.95% | 99.17% | **99.43%** | 97.97% | 74.41% | 99.17% | 97.25% | 98.89% |
| Recall (Cracked) | 97.22% | 99.17% | 96.39% | 93.61% | 87.22% | **99.44%** | 98.33% | 98.89% |
| F1 (Cracked) | 97.58% | 99.17% | 97.88% | 95.74% | 80.31% | **99.31%** | 97.79% | 98.89% |

**Our 1D CNN is the best overall model**, edging out the SVM on accuracy, recall, and F1 — with zero hand-crafted features. Caveats vs. the paper:
- Paper used 70/30 split; we used 80/20 (more training data = slight advantage)
- Paper used Equal Size Sampling (ESS) to balance classes; we used `class_weight=balanced`
- Paper tuned SVM hyperparameters with Bayesian optimization (TPE); we used C=10, gamma='scale'
- Our `Class_sounds/` data only contains C02 and C04 positions (closest to crack, easiest to classify) — the paper also tested farther positions

### Our 5-Fold Cross-Validation Results (tabular classifiers)

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
|-------|-------------|--------------|-----------|-------|
| **SVM** | **99.78%** | **99.85%** | **99.88%** | **99.87%** |
| XGBoost | 99.26% | 99.18% | 99.93% | 99.55% |
| Random Forest | 98.73% | 98.69% | 99.78% | 99.23% |

*Neural networks (CNN_MFCC, CNN_1D) use best validation metrics across training epochs rather than 5-fold CV (too expensive on CPU).*

**Class imbalance:** Cracked (1,800) vs Intact (8,236) — ~4.6x ratio. Handled via `class_weight=balanced` (RF, SVM), `scale_pos_weight=4.58` (XGBoost), and weighted `CrossEntropyLoss` (CNNs).

### Best Classifier Per Metric (our models, all 7)

| Metric | Winner | Score |
|--------|--------|-------|
| Accuracy | 1D CNN | 99.75% |
| Precision (Cracked) | XGBoost | 99.43% |
| Recall (Cracked) | 1D CNN | 99.44% |
| F1 (Cracked) | 1D CNN | 99.31% |

Comparison plots saved to `Classifier Results/Comparison/`.

### MFCC CNN Results & Findings

The CNN (EfficientNet-B0, frozen ImageNet weights) achieved **92.33% accuracy** — lower than all other models. Key reasons:

- **Domain mismatch:** EfficientNet was pretrained on natural photographs (ImageNet). MFCC spectrograms look nothing like natural images, so the learned features don't transfer as effectively as they would in a vision task.
- **Frozen backbone:** No fine-tuning of the backbone weights means the network never adapts to the acoustic domain. Full fine-tuning on a GPU would likely close this gap.
- **Hand-crafted features win on small specialized data:** The 29 domain-specific features (Df, Vf, MFCC statistics) encode expert acoustic knowledge directly. For small datasets like ours, this prior knowledge outperforms generic deep features.
- **Recall is respectable (87.22%):** The model correctly flags 87% of cracked samples. The weaker metric is precision (74.41%), meaning it generates more false positives than the classical models.

### 1D CNN Results & Findings

The 1D CNN trained directly on raw waveforms achieved **99.75% accuracy, 99.31% F1** — the best of all models despite using no feature engineering.

- **End-to-end learning works:** The network learned its own filters from raw audio, capturing acoustic structure directly without any domain knowledge encoded in the features.
- **Matches classical ML without hand-crafted features:** Competing with a carefully tuned SVM on 29 domain-specific features demonstrates the raw waveform contains sufficient discriminating information.
- **Training instability at epoch 15:** Val loss spiked to 4.06 and F1 collapsed to 0.19 before the LR scheduler halved the rate and recovery was swift. Best checkpoint was epoch 43 (val F1=1.00 on the 180-sample val set).
- **~478K parameters, trained from scratch on CPU** in under 30 minutes — lightweight and practical.

---

## Level 2a — Crack Width Classification Results

**Dataset:** 1,800 cracked WAV files across 3 width classes (600 per class). 9 specimen/position folders total.

**Critical finding — data leakage with random splits:** With a standard random 80/20 train/test split, all models achieve ~99-100% accuracy because each specimen contributes ~200 nearly-identical recordings. When recordings from the same specimen appear in both train and test, the model learns to recognize the *specimen's acoustic fingerprint* rather than the physics of crack width. This produces inflated metrics that do not reflect real-world generalization ability.

**Specimen-level split (correct methodology):** Series 1 specimens (`specimen_1_X`, both C02 and C04 positions, 400 recordings/class) → train. Series 2 specimens (`specimen_2_X`, C02 only, 200 recordings/class) → test. The two series are physically different concrete specimens — the model must generalize to a specimen it has never seen.

### Results with Specimen-Level Split

| Model | Test Accuracy | Test Macro F1 | CV Accuracy (within series 1) | CV Macro F1 |
|-------|--------------|---------------|-------------------------------|-------------|
| Random Forest | 34.5% | 35.1% | 99.75% | 99.75% |
| XGBoost | 41.8% | 38.2% | 99.67% | 99.67% |
| SVM | 33.5% | 17.0% | 99.83% | 99.83% |
| MLP | 33.2% | 19.2% | ~99% (val F1=1.00) | — |

The stark contrast between CV (within series 1, ~99%) and test (series 2, 33-42%) confirms that the models are memorizing specimen-specific acoustic signatures rather than learning generalizable width features. The 33% test accuracy is near random chance (33.3% for a 3-class problem), meaning the models learn nothing that transfers to an unseen specimen.

**Interpretation:** Crack width may be fundamentally difficult to distinguish from acoustic features alone when only 2 physical specimens (one per series) are available per width class. The tabular features (Df, Vf, MFCC statistics) that work well for crack *detection* (Cracked vs Intact, 9 distinct specimens) do not generalize for crack *width* classification when specimens are few and highly varied between series. The paper's 99.44% width result likely reflects a similar random-split leakage issue.

---

## Limitations & Cross-Dataset Generalization

### Domain Shift — Why Models Don't Transfer Out-of-the-Box

All classifiers in this pipeline were trained and tested on audio from the **same six concrete specimens** recorded in the same controlled laboratory environment (same microphone, same foam-lined box, same hammer). This means models learn a mixture of:

1. Genuine acoustic physics — resonance frequency shifts caused by cracks
2. Specimen-specific fingerprints — the unique frequency response of each particular block of concrete, shaped by its exact mix, geometry, aggregate distribution, and surface texture

When you try to apply these models to **different concrete samples** (e.g., from a university lab), the specimen fingerprints don't match. The model confidently classifies based on patterns it learned from the original six specimens, which are irrelevant to the new samples. The result is near-random accuracy on the new data even though the model performed at 99%+ on the original test set.

This is a well-known problem in structural health monitoring called **domain shift** — and it applies to any model trained on a public dataset from specific specimens.

### The Fine-Tuning Solution

The standard fix is **fine-tuning**: take the pre-trained model and continue training it on a small number of labeled samples from the target environment (your lab, your specimens, your microphone). This adapts the model's learned representations to the new acoustic domain.

For crack detection (Level 1), fine-tuning is tractable: the crack vs. intact signal is strong and binary, so even a modest number of labeled samples (tens to hundreds) from the new specimens would likely be sufficient for the model to recalibrate.

**In our case, we were not able to collect sufficient labeled samples from the university lab** to perform fine-tuning, which meant cross-dataset transfer remained poor. This is a practical data collection bottleneck, not a fundamental flaw in the approach.

### Key Takeaways

- High accuracy on the paper's dataset does **not** imply high accuracy on unseen specimens from a different lab.
- The more physically diverse the training specimens (different concrete mixes, different labs, different recording setups), the more generalizable the model.
- Fine-tuning with even a small number of labeled samples from the target environment is the most direct path to deployment.
- The crack width problem (Section above) is the same issue at a smaller scale — models memorized the two physical specimens per width class rather than learning width physics.

---

## Next Steps

### Neural Networks
- [x] **CNN on MFCC images** — EfficientNet-B0 feature extraction + MLP head. 92.33% accuracy. Lower than classical ML due to ImageNet domain mismatch and frozen backbone. Full fine-tuning on GPU would close the gap.
- [x] **1D CNN on raw waveform** — fully end-to-end, no feature engineering. 99.75% accuracy, 99.31% F1. Best overall model — matches SVM without any hand-crafted features.
- [x] **LSTM on MFCC frame sequences** — 9 frames × 13 coefficients per hit. 99.20% accuracy, 97.79% F1. Competitive with XGBoost without hand-crafted features.
- [x] **MLP on tabular features** — same 29 features as classical classifiers, different learner. 99.60% accuracy, 98.89% F1. Outperforms XGBoost and LSTM; confirms the features are rich enough for a small NN to exploit.

### Classification Extensions
- [x] **Crack width classification** — Level 2a: 3-class (0.2 / 0.4 / 0.6 mm). Specimen-level split reveals ~33-42% test accuracy — near-random chance on an unseen specimen. See Level 2a results above.
- [ ] **Crack depth classification** — Level 2b from the paper: binary (20 mm / 40 mm)
- [ ] **Align MFCC features with paper** — re-extract using 12 coefficients (drop #1), max + mean per coefficient (24 features) for a fairer apples-to-apples comparison
