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

**Hammering positions:** Lines were drawn at 2 cm intervals perpendicular to the crack. `C02`, `C04`, `C06`... = positions on the **cracked side** at 2 cm, 4 cm, 6 cm... from the crack. `N02`, `N04`... = same distances on the **intact side**. Our `Class_sounds/` folder only contains the two closest cracked positions (C02, C04) and intact specimens.

**Recording setup:** Sony ECM-PCV80U condenser microphone inside a foam-lined cardboard box (500 × 400 × 40 mm high-density foam) to isolate ambient noise. 200 hammer hits recorded per position per specimen.

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
  Random Forest | XGBoost | SVM
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
Output folder for all ML results, organized by model and comparison.

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

### `Classifier_Comparison.py`
Loads `metrics.json` from each classifier's output folder and produces side-by-side comparison plots in `Classifier Results/Comparison/`. **Run after all five classifiers.**

---

## Our Results vs. Paper (Crack Detection)

| | Paper SVM | Our SVM | Our XGBoost | Our Random Forest | Our MFCC CNN | Our 1D CNN |
|---|---|---|---|---|---|---|
| Accuracy | 97.59% | 99.70% | 99.25% | 98.51% | 92.33% | **99.75%** |
| Precision (Cracked) | 97.95% | 99.17% | **99.43%** | 97.97% | 74.41% | 99.17% |
| Recall (Cracked) | 97.22% | 99.17% | 96.39% | 93.61% | 87.22% | **99.44%** |
| F1 (Cracked) | 97.58% | 99.17% | 97.88% | 95.74% | 80.31% | **99.31%** |

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

### Best Classifier Per Metric (our models)

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

## Next Steps

### Neural Networks (in progress)
- [x] **CNN on MFCC images** — EfficientNet-B0 feature extraction + MLP head. 92.33% accuracy. Lower than classical ML due to ImageNet domain mismatch and frozen backbone. Full fine-tuning on GPU would close the gap.
- [x] **1D CNN on raw waveform** — fully end-to-end, no feature engineering. 99.75% accuracy, 99.31% F1. Best overall model — matches SVM without any hand-crafted features.
- [ ] **LSTM on MFCC frame sequences** — treat each hit as a time series of MFCC frames (~15-20 frames × 13 coefficients). LSTM captures temporal evolution of hammer strike (attack → resonance → decay) that mean/std features throw away.
- [ ] **MLP on tabular features** — simple NN baseline on the same 29 features used by classical classifiers.
- [ ] **Autoencoder anomaly detection** — train only on Intact sounds; cracked = high reconstruction error. Unsupervised; treats class imbalance as an advantage.

### Classification Extensions
- [ ] **Crack width classification** — Level 2a from the paper: 3-class problem (0.2 / 0.4 / 0.6 mm) using specimen subfolders already in `Class_sounds/Cracked/`
- [ ] **Crack depth classification** — Level 2b from the paper: binary (20 mm / 40 mm) using series 1 vs series 2 specimens
- [ ] **Align MFCC features with paper** — re-extract using 12 coefficients (drop #1), max + mean per coefficient (24 features) for a fairer apples-to-apples comparison
