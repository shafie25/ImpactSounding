# Technical Notes — Acoustic Impact Hammer Testing Pipeline

This document covers the theory, implementation details, and parameter choices behind the pipeline.

**Attribution note:** Sections 1–3 describe the original paper's experimental design and preprocessing. Our contribution begins at Section 4 (feature extraction re-implementation) and Section 5 onward (all ML models).

> Paper: *"Artificial intelligence enhanced automatic identification for concrete cracks using acoustic impact hammer testing"* — Alhebrawi, Huang, Wu (2023). DOI: 10.1007/s13349-022-00651-8

---

## 1. Dataset & Experimental Setup (Original Paper)

The paper's authors fabricated six reinforced concrete blocks (400 × 100 × 100 mm, compressive strength ~29 MPa). Artificial cracks were created by embedding greased polyethylene sheets at 45° before casting and removing them after curing — giving controlled, repeatable crack geometries.

| Specimen | Width | Depth |
|----------|-------|-------|
| 1.1 | 0.2 mm | 40 mm |
| 1.2 | 0.4 mm | 40 mm |
| 1.3 | 0.6 mm | 40 mm |
| 2.1 | 0.2 mm | 20 mm |
| 2.2 | 0.4 mm | 20 mm |
| 2.3 | 0.6 mm | 20 mm |

**Hammering positions:** Lines were drawn at 2 cm intervals perpendicular to the crack. Positions are labeled by side and distance: `C02`, `C04`, ... = cracked side at 2, 4 cm from crack. `N02`, `N04`, ... = intact side. **200 hammer hits recorded per position per specimen.**

**Recording setup:** Sony ECM-PCV80U condenser microphone inside a foam-lined box (500 × 400 × 40 mm foam) to isolate ambient noise.

**What we received:** The paper provided us with pre-segmented, pre-labeled WAV files already organized into `Specimens/Class_sounds/Cracked/` and `Specimens/Class_sounds/Intact/`. Our dataset contains only the **two closest cracked positions (C02, C04)** plus all intact positions — 10,036 WAV files total.

---

## 2. Audio Preprocessing (Original Paper — Not Performed by Us)

> **We did not perform this step and did not have access to the raw recordings.** We received the data already preprocessed: 10,036 individual WAV files, one per hammer hit, already segmented, trimmed, and organized into `Specimens/Class_sounds/Cracked/` and `Specimens/Class_sounds/Intact/`. The script in `Original Paper Files/Hammer Impact Detection & Segmentation.py` is the paper's code — it is not used anywhere in our pipeline.
>
> This section is included only to document what the paper's team did upstream so the format of the WAV files we work with makes sense.

### What the paper's team did

Raw recordings contained many hammer hits back to back. They isolated each individual hit using silence-based segmentation, then trimmed each to **0.20 s** to capture the strike, direct wave propagation, and early resonance. Hits below a peak amplitude threshold were discarded as noise. All audio was resampled to **22,050 Hz**.

The result — which is what we received — is a set of short, fixed-length, single-hit WAV files at 22,050 Hz, each approximately 4,410 samples long.

---

## 3. Feature Concepts from the Paper

The paper identified three feature types for acoustic crack detection. We re-implemented all three in our own extraction script. The mathematical definitions below follow the paper; the implementation is ours.

### 3.1 Dominant Frequency — Df

The **Fast Fourier Transform (FFT)** decomposes a time-domain signal into its frequency components. For a discrete signal `x[n]` of length `N`:

```
X[k] = sum_{n=0}^{N-1} x[n] * e^{-j2πkn/N}
```

`|X[k]|` is the amplitude at frequency bin `k`. The dominant frequency **Df** is the frequency at which this magnitude is highest:

```
Df = freq_axis[ argmax(|X[k]|) ]
```

**Physical meaning:** A cracked block has lower structural stiffness than an intact one — lower stiffness produces lower resonant frequencies. So cracked hits tend to shift the dominant frequency downward. Df is the simplest discriminating feature but cannot alone distinguish crack width or depth.

**Paper's finding:** Df can detect crack presence but cannot reliably distinguish width or depth.

### 3.2 Frequency Feature Value — Vf

Df only tells you where the spectral peak is. Vf captures the **spread of energy** across all frequencies — essentially the standard deviation of the spectrum weighted by energy:

```
f_bar = sum(|X[k]|^2 * f[k]) / sum(|X[k]|^2)      [energy-weighted mean frequency]

Vf = sqrt( sum(|X[k]|^2 * (f[k] - f_bar)^2) ) / sqrt( sum(|X[k]|^2) )
```

**Physical meaning:** A crack changes not just the peak frequency but how energy is distributed across frequencies — cracked surfaces scatter energy differently than intact ones.

**Paper's finding:** Vf works better for shallow (20 mm) cracks but fails for deep (40 mm) cracks.

### 3.3 Mel-Frequency Cepstral Coefficients — MFCC

MFCCs are the best-performing features in this pipeline. The paper used them with `n_mfcc=13`, dropping coefficient #1, and extracting **max and mean** per coefficient (24 features total). We re-implemented this with a slight variation (see Section 4.1).

The extraction pipeline has four steps:

**Step 1 — Short-Time Fourier Transform (STFT)**
The signal is divided into short overlapping chunks and FFT is applied to each, producing a 2D time-frequency spectrogram. (librosa defaults: `n_fft=2048`, `hop_length=512`.)

**Step 2 — Mel filterbank**
Human hearing perceives frequency on a non-linear scale — differences at low frequencies matter more than the same differences at high frequencies. The **mel scale** approximates this:

```
mel(f) = 2595 * log10(1 + f / 700)
```

A bank of triangular filters is applied to the spectrogram, converting from linear Hz to the mel scale. This compresses high-frequency detail and expands low-frequency detail.

**Step 3 — Log compression**
The log of each filter's energy is taken, compressing dynamic range and improving robustness to amplitude variation between hits.

**Step 4 — Discrete Cosine Transform (DCT)**
The DCT decorrelates the log-mel energies and compacts them into a small number of coefficients. The first `n_mfcc` outputs are kept — each represents a different aspect of the spectral shape.

**Result:** A matrix of shape `(n_mfcc, T)` — coefficients × time frames. Each row shows how one coefficient evolves during the hit.

**Paper's finding:** MFCCs with an SVM classifier achieved the best results across all three classification tasks (crack detection, width, depth).

---

## 4. Our Feature Extraction (`Labeled Feature Extraction.py`)

We re-implemented all three feature types from the paper and applied them to the full 10,036 WAV dataset, producing `Labeled_Features.xlsx`.

### What we did differently from the paper

| | Paper | Our implementation |
|---|---|---|
| MFCC coefficients | 12 (drops #1) | 13 (keeps all) |
| MFCC statistics | max + mean per coefficient | **mean + std** per coefficient |
| Total MFCC features | 24 | 26 |
| Applied to | Subset of data | All 10,036 WAVs |

**Why mean + std instead of max + mean:** Both are valid summarizations. Mean captures the typical spectral envelope; std captures the variability/dynamics of each coefficient during the hit. Max captures the peak behavior. We chose mean + std to represent both the average state and how much it fluctuates — a physically reasonable choice for a short transient signal.

### Features per file (29 total)

| Feature | Count | Description |
|---------|-------|-------------|
| `Df_Hz` | 1 | FFT peak frequency (Hz) |
| `Df_Amplitude` | 1 | Magnitude at peak frequency |
| `Vf` | 1 | Energy-weighted frequency spread |
| `MFCC_k_mean` | 13 | Mean of each coefficient over all time frames |
| `MFCC_k_std` | 13 | Std of each coefficient over all time frames |

---

## 5. Our Classical ML Models

All three models are trained on `Labeled_Features.xlsx` with an **80/20 stratified train/test split** (random state = 42). Class imbalance: Cracked (1,800) vs Intact (8,236), ~4.6:1.

### 5.1 Random Forest (`Classifier_RandomForest.py`)

A **Random Forest** is an ensemble of decision trees. Each tree is trained on a random bootstrap sample of the data, and at each split only a random subset of features is considered. Final predictions are majority vote across all trees. Decorrelating the trees this way means individual errors average out.

**Our parameters:**
- `n_estimators = 300` — 300 trees; more trees = more stable predictions (diminishing returns past ~200)
- `class_weight = 'balanced'` — automatically upweights Cracked by ~4.6× in the loss so the model doesn't ignore the minority class

**Result:** 98.51% test accuracy, 95.74% F1 on Cracked.

---

### 5.2 XGBoost (`Classifier_XGBoost.py`)

**Gradient Boosted Trees** build an ensemble sequentially — each new tree is fit to the residual errors of the current ensemble. **XGBoost** is an optimized implementation with L1/L2 regularization and faster training.

**Our parameters:**
- `scale_pos_weight = 4.58` — XGBoost's equivalent of `class_weight='balanced'`. Set to `n_intact / n_cracked = 8236 / 1800 ≈ 4.58`. Tells the model to treat each Cracked sample as if it appeared 4.58× more often.
- All other parameters: XGBoost defaults

**Result:** 99.25% test accuracy, 97.88% F1. Best precision of all models (99.43%).

---

### 5.3 SVM (`Classifier_SVM.py`)

An SVM finds the **maximum-margin hyperplane** separating the two classes — the decision boundary that is as far as possible from the nearest training points (support vectors). Maximizing the margin improves generalization to unseen data.

**The kernel trick:** Our 29-dimensional data is not linearly separable. The **RBF kernel** implicitly maps data to a higher-dimensional space where it may be separable, without computing that mapping explicitly:

```
K(x, x') = exp(-gamma * ||x - x'||^2)
```

This measures similarity: 1 when samples are identical, near 0 when far apart.

**Our parameters:**
- `C = 10` — regularization strength. Higher C = tighter fit to training data. C=10 is moderately strict.
- `gamma = 'scale'` — automatically set to `1 / (n_features × X.var())`. Scales the kernel width to the data's variance.
- `class_weight = 'balanced'` — same imbalance handling as RF
- **StandardScaler applied first** — SVM is sensitive to feature scale because the RBF kernel uses Euclidean distance. Without scaling, features with larger numeric ranges would dominate the kernel.

**Result:** 99.70% test accuracy, 99.17% F1 — best overall. Consistent with the paper's finding that SVM is the top performer.

---

## 6. Our CNN — EfficientNet-B0 on MFCC Spectrograms (`Classifier_CNN_MFCC.py`)

This is entirely our own addition — not from the paper.

### 6.1 MFCC Spectrogram Images (`MFCC_Image_Generator.py`)

Instead of collapsing MFCC matrices to 26 summary statistics, we convert each hit to a **2D image** — coefficients (rows) × time frames (columns), color-mapped. Each pixel encodes one coefficient at one time step, preserving the full temporal structure that mean/std throws away.

We generated 10,036 images using `MFCC_Image_Generator.py` with the following process per WAV file:

1. Load audio at native sample rate (`sr=None`, 22,050 Hz) — same as tabular extraction
2. Compute MFCC matrix with `librosa.feature.mfcc(n_mfcc=13)` — produces a `(13, T)` matrix
3. Create a matplotlib figure sized exactly to fill 224×224 pixels (`figsize=(2.24, 2.24)`, `dpi=100`), with the axes expanded to cover the entire figure (no margins, no labels, no colorbar)
4. Render the matrix as a color heatmap using `librosa.display.specshow` — rows are the 13 MFCC coefficients, columns are time frames, color intensity encodes coefficient value
5. Save as JPG at 100 DPI

**Why 224×224?** This is the standard input size for ImageNet-pretrained models (ResNet, EfficientNet, etc.), so no resizing is needed at training time.

**Why no axes or labels?** The CNN learns from pixel patterns, not from axis annotations. Including them would add non-informative structure that the model might spuriously learn from.

Output: `MFCC_Images/Cracked/` and `MFCC_Images/Intact/` — 1,800 and 8,236 images respectively.

> **Note:** The `Plots/` subfolders already inside each specimen folder are **waveform plots** (time-domain amplitude), not spectrograms. They were generated by the paper's segmentation script for visual inspection only and are not used in our CNN pipeline.

### 6.2 EfficientNet-B0 Architecture

EfficientNet-B0 (Tan & Le, Google, 2019) is a CNN that jointly scales network depth, width, and input resolution using a compound scaling formula, yielding better accuracy per parameter than older architectures like ResNet.

It is built from stacked **MBConv (Mobile Inverted Bottleneck Convolution)** blocks. Each block:
1. **Expands** channels with a 1×1 convolution
2. **Depthwise convolution** — one filter per channel (much cheaper than standard convolution)
3. **Squeeze-and-Excitation** — learns to re-weight channels by importance
4. **Projects** back down with a 1×1 convolution

After all blocks, **Global Average Pooling** reduces the spatial dimensions to a **1,280-dimensional feature vector** per image. This is the embedding we use for classification.

**Pretrained on ImageNet** — 1.2M photographs, 1,000 categories. ~5.3M total parameters.

### 6.3 Our Transfer Learning Strategy

We used **feature extraction** rather than full fine-tuning, due to CPU constraints:

1. Freeze all backbone weights (ImageNet weights locked, no gradients)
2. Replace the classifier head with: `Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→2)`
3. **Pre-extract features:** run all 10,036 images through the frozen backbone once (~4 min) → cache 1,280-dim vectors
4. **Train only the head** on cached vectors for 50 epochs (~seconds per epoch)

**Why not full fine-tuning?** Fine-tuning requires a forward + backward pass through the full backbone every epoch. On CPU with 10k images, one epoch takes ~20 minutes. By caching features, the backbone runs once and training the head is near-instant.

**Split:** 80/10/10 stratified train/val/test.

**Class weighting:**
```
weight[Cracked] = N / (2 × n_cracked) = 10036 / 3600  = 2.788
weight[Intact]  = N / (2 × n_intact)  = 10036 / 16472 = 0.609
```

### 6.4 Why the CNN Underperformed

The CNN achieved **92.33% accuracy, 80.31% F1** — the lowest of all models.

- **Domain mismatch:** EfficientNet was pretrained on natural photographs. MFCC spectrograms look nothing like natural images, so ImageNet features don't transfer as effectively as they would on a vision task.
- **Frozen backbone = no adaptation:** The convolutional filters never update to the acoustic domain. Full fine-tuning on a GPU would allow the network to adapt and would likely improve results significantly.
- **Hand-crafted features encode domain knowledge:** The 29 tabular features were designed specifically for acoustic crack detection. On small specialized datasets, this kind of prior knowledge tends to outperform generic deep features.
- **Recall is still reasonable (87.22%):** The model catches most actual cracks. The weaker metric is precision (74.41%) — more false alarms than the classical models.

---

## 7. Evaluation Methodology (Our Choices)

### Why accuracy alone is misleading

Our dataset has 8,236 Intact vs 1,800 Cracked (~4.6:1). A model that predicts "Intact" for everything would get **82.1% accuracy** while being completely useless. This is why we always report precision, recall, and F1.

### Precision vs Recall

```
Precision = TP / (TP + FP)   [of everything flagged as cracked, how many actually are?]
Recall    = TP / (TP + FN)   [of all actual cracks, how many did we catch?]
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

For structural health monitoring, **recall matters more** — a missed crack (false negative) is more dangerous than a false alarm (false positive). All values are reported for the Cracked class.

### Train/Test Split

**80/20 stratified** (random state = 42). Stratified = Cracked:Intact ratio preserved in both sets. The paper used 70/30 — our 80/20 gives the model slightly more training data, partially explaining why our SVM outperforms theirs.

### 5-Fold Cross-Validation

The training set is split into 5 equal folds. The model trains on 4 and evaluates on 1, rotating through all 5 combinations. Averaging results gives a more reliable performance estimate than a single split and helps detect overfitting.

Applied to RF, XGBoost, and SVM only. For the neural networks, 5× training cost is impractical on CPU — we report best validation metrics across epochs instead.

---

## 8. Our 1D CNN — Raw Waveform Classifier (`Classifier_1DCNN.py`)

This is entirely our own addition — not from the paper.

### 8.1 Motivation

The tabular classifiers and MFCC CNN both rely on feature extraction — either hand-crafted (Df, Vf, MFCC statistics) or learned from images (EfficientNet). A 1D CNN on raw audio is **fully end-to-end**: it receives the raw amplitude samples directly and learns its own filters without any domain knowledge encoded in advance.

This approach answers the question: *does the raw waveform contain enough discriminating information, and can a small network find it without feature engineering?*

### 8.2 Input Representation

Each WAV file is loaded at 22,050 Hz and truncated/padded to exactly **4,410 samples** (0.20 s). The signal is treated as a 1D time series with shape `(1, 4410)` — one channel, 4,410 time steps. No STFT, no mel filterbank, no FFT — just raw amplitude values.

### 8.3 Data Augmentation

Applied to the training set only (not val/test):

- **Gaussian noise:** adds random perturbation with std = 0.5% of the signal's own std. Mimics microphone noise variation between hits.
- **Random circular time shift:** shifts the waveform by a random amount up to ±5% of its length (±220 samples). Mimics variation in when exactly the hammer strike lands within the recording window.

Both augmentations are physically plausible for impact hammer recordings and prevent the model from overfitting exact sample positions.

### 8.4 Architecture

```
Input: (B, 1, 4410)

Conv1d(1→32,   kernel=64, stride=4) + BatchNorm + ReLU   → (B, 32,  ~1087)
Conv1d(32→64,  kernel=32, stride=2) + BatchNorm + ReLU   → (B, 64,  ~528)
Conv1d(64→128, kernel=16, stride=2) + BatchNorm + ReLU   → (B, 128, ~257)
Conv1d(128→256, kernel=8, stride=2) + BatchNorm + ReLU   → (B, 256, ~125)

AdaptiveAvgPool1d(1)   →  (B, 256, 1)
Flatten                →  (B, 256)
Linear(256→64) + ReLU
Dropout(0.3)
Linear(64→2)
```

**Total parameters: ~478,000** — lightweight, trained from scratch on CPU.

**Design rationale:**
- **Large first kernel (64 samples ≈ 3 ms):** captures broad temporal patterns including the initial attack transient, which spans several milliseconds at 22,050 Hz.
- **Stride instead of pooling:** reduces sequence length while learning which parts of the signal to compress, rather than applying a fixed pooling operation.
- **Increasing channel depth (1→32→64→128→256):** progressively abstracts from raw amplitudes to higher-level acoustic features.
- **BatchNorm after each conv:** stabilizes training, reduces sensitivity to initialization, and acts as mild regularization.
- **Global Average Pooling:** collapses the entire time dimension into a single 256-dim vector per sample, making the classifier independent of exact sequence length and robust to slight timing variations.

### 8.5 Training Setup

| Parameter | Value |
|-----------|-------|
| Split | 80/10/10 stratified (train/val/test) |
| Epochs | 50 |
| Optimizer | Adam, lr=1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | CrossEntropyLoss with class weights |
| Class weights | Cracked=2.788, Intact=0.609 |
| Batch size | 64 |
| Best checkpoint | Saved by val F1 (Cracked class) |

**Class weights formula** (same as CNN_MFCC):
```
weight[Cracked] = N / (2 × n_cracked) = 10036 / 3600  = 2.788
weight[Intact]  = N / (2 × n_intact)  = 10036 / 16472 = 0.609
```

### 8.6 Results

**Test set:** 99.75% accuracy, 99.17% precision, 99.44% recall, 99.31% F1 — the best of all our models.

**Training notes:** Val loss spiked to 4.06 at epoch 15 and F1 collapsed to 0.19, after which the LR scheduler halved the rate and recovery was rapid. Best checkpoint: epoch 43 (val F1=1.000 on the 180-sample val set). This instability is typical when learning rate is too high for the current loss landscape — the scheduler handled it automatically.

### 8.7 Why the 1D CNN Outperformed the MFCC CNN

| | MFCC CNN (EfficientNet-B0) | 1D CNN |
|--|--|--|
| Input | 224×224 MFCC spectrogram image | Raw 4,410-sample waveform |
| Backbone | Pretrained ImageNet weights (frozen) | Trained from scratch |
| Domain match | Poor (photos ≠ spectrograms) | Perfect (filters learn acoustic patterns directly) |
| Accuracy | 92.33% | 99.75% |
| F1 (Cracked) | 80.31% | 99.31% |

The key difference is domain match. EfficientNet's ImageNet filters were learned on natural photographs and don't adapt (backbone is frozen). The 1D CNN's filters were learned entirely from the acoustic data — they encode whatever waveform patterns actually distinguish cracked from intact blocks.

### 8.8 Why the 1D CNN Matches Classical ML

The SVM and XGBoost were given 29 hand-crafted features encoding known acoustic discriminators (dominant frequency, frequency spread, MFCC statistics). The 1D CNN was given nothing but raw samples — and achieved essentially the same performance. This suggests:

1. The raw waveform contains the same (or more) discriminating information as the hand-crafted features.
2. With ~10,000 training examples, even a modest convolutional network can discover those discriminators without prior domain knowledge.
3. For larger datasets or harder tasks (width/depth classification), end-to-end learning would likely pull further ahead.

---

## 9. Our LSTM — MFCC Frame Sequence Classifier (`Classifier_LSTM.py`)

### 9.1 Motivation

The tabular classifiers (RF, XGBoost, SVM, MLP) all collapse the MFCC matrix to 26 summary statistics — mean and std per coefficient. This discards the temporal dimension: *when* things happen during the hit is thrown away. A hammer impact on concrete has distinct temporal phases — the sharp attack transient, the direct wave propagation, the resonance build-up, and the exponential decay — and cracked vs intact blocks differ in how this evolution unfolds, not just in its average.

The LSTM operates on the full MFCC frame sequence, giving the model access to this temporal information directly.

### 9.2 Input Representation

Each WAV file is loaded and converted to an MFCC sequence:

1. Pad/truncate to exactly 4,410 samples (0.20 s at 22,050 Hz)
2. Compute `librosa.feature.mfcc(n_mfcc=13, hop_length=512, n_fft=2048)`
3. Transpose from `(13, T)` to `(T, 13)` — time is the sequence dimension

With `hop_length=512` and `center=True` (librosa default), the number of frames is:
```
T = 1 + 4410 // 512 = 9 frames
```
All files produce exactly 9 frames — no padding needed. Each frame has 13 MFCC coefficients. Input shape per sample: `(9, 13)`.

### 9.3 Data Augmentation

Applied to training only:
- **Gaussian noise on MFCC values:** std = 0.5% of the feature's own std. Mimics minor variation in recording conditions.
- **Time masking:** one randomly chosen frame is zeroed out. Inspired by SpecAugment; prevents the model from relying on any single time step.

### 9.4 Architecture

```
Input: (B, 9, 13)

LSTM(input=13, hidden=64, num_layers=2, dropout=0.3, batch_first=True)
→ last hidden state of top layer: (B, 64)

Linear(64→32) + ReLU + Dropout(0.3)
Linear(32→2)

Output: (B, 2) logits
```

**Why the last hidden state (not mean pooling)?** In a 0.20s hammer strike, the decay phase (final frames) carries the most information about structural stiffness — cracked blocks decay differently. The last hidden state gives the most weight to recent context.

**Total parameters: ~55,650** — lightweight, trains in ~10 minutes on CPU.

### 9.5 Training Setup

| Parameter | Value |
|-----------|-------|
| Split | 80/10/10 stratified (train/val/test) |
| Epochs | 50 |
| Optimizer | Adam, lr=1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | CrossEntropyLoss with class weights |
| Class weights | Cracked=2.788, Intact=0.609 |
| Batch size | 64 |
| Best checkpoint | Saved by val F1 (Cracked class) |

### 9.6 Results

**Test set:** 99.20% accuracy, 97.25% precision, 98.33% recall, 97.79% F1 (Cracked class). Best checkpoint: epoch 37.

**Interpretation:** The LSTM matches XGBoost (97.88% F1) without any hand-crafted features — it derives comparable discriminating power purely from the 9-frame MFCC sequence. It falls short of the 1D CNN (99.31% F1), which operates on the full 4,410-sample waveform and has access to much finer temporal resolution. The 9-frame sequence is a compressed representation; at this level of compression, the LSTM has limited advantage over the tabular approaches that summarize the same frames as mean/std.

---

## 10. Our MLP — Tabular Feature Classifier (`Classifier_MLP.py`)

### 10.1 Motivation

The MLP provides the fairest neural network vs. classical ML comparison: it uses exactly the same 29 features as SVM, XGBoost, and Random Forest — but a different learner. If the MLP underperforms the SVM, the feature space is likely already (near-)linearly separable with a suitable kernel. If it matches or exceeds, it shows that learned nonlinear combinations of the same features add discriminating power.

### 10.2 Feature Normalization

Unlike tree-based models, MLPs are sensitive to feature scale. `StandardScaler` (zero mean, unit variance) is fit on the training set and applied to val and test — the standard way to avoid data leakage while ensuring consistent scaling.

The 29 features span very different numeric ranges: `Df_Hz` is in the hundreds of Hz, `Vf` is also a frequency-scale quantity, and MFCC values are dimensionless dB-like numbers. Without scaling, features with larger ranges dominate the gradient updates in the first layer.

### 10.3 Architecture

```
Input: (B, 29)

Linear(29→128) + BatchNorm1d(128) + ReLU + Dropout(0.3)
Linear(128→64) + BatchNorm1d(64)  + ReLU + Dropout(0.3)
Linear(64→2)

Output: (B, 2) logits
```

**BatchNorm on tabular data:** Even after StandardScaler, per-feature normalization doesn't account for inter-batch distributional shift. BatchNorm normalizes each hidden layer's activations, stabilizing training and reducing sensitivity to the learning rate.

**Total parameters: ~12,610** — the smallest model in the pipeline by a wide margin.

### 10.4 Training Setup

| Parameter | Value |
|-----------|-------|
| Split | 80/10/10 stratified (train/val/test) |
| Epochs | 100 (fast — no audio I/O, tiny model) |
| Optimizer | Adam, lr=1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Loss | CrossEntropyLoss with class weights |
| Class weights | Cracked=2.788, Intact=0.609 |
| Batch size | 64 |
| Best checkpoint | Saved by val F1 (Cracked class) |

### 10.5 Results

**Test set:** 99.60% accuracy, 98.89% precision, 98.89% recall, 98.89% F1 (Cracked class). Best checkpoint: epoch 53.

**Interpretation:** The MLP outperforms XGBoost (97.88% F1) and the LSTM (97.79% F1) on the same feature set, and comes close to the SVM (99.17% F1). This tells us:
- The 29 tabular features contain enough information for a small MLP to extract strong classification signals.
- The feature space is not perfectly linearly separable — the MLP's nonlinearity adds value over a purely linear model on the same input.
- The SVM's RBF kernel still has a slight edge, suggesting the optimal decision boundary has structure that a 2-layer MLP approximates well but not perfectly.
- Training on CPU takes under 2 minutes — the most practical NN in the pipeline.

---

## 11. Model Visualization (`visualize_models.py`)

### 11.1 Architecture Summaries (torchinfo)

`torchinfo` wraps PyTorch's built-in `__repr__` with a forward-pass trace to report the exact input and output tensor shape at every layer, along with per-layer and total parameter counts. This is more useful than `print(model)` because it shows how the temporal dimension shrinks through the conv stack.

Example for CNN_1D — the time axis collapses from 4,410 to ~125 before Global Average Pooling:

```
Conv1d(1→32,   k=64, s=4)  input (1,1,4410)  →  output (1,32,1103)
Conv1d(32→64,  k=32, s=2)                    →  output (1,64,548)
Conv1d(64→128, k=16, s=2)                    →  output (1,128,272)
Conv1d(128→256, k=8, s=2)                    →  output (1,256,134)
AdaptiveAvgPool1d(1)                          →  output (1,256,1)
Linear(256→64)                                →  output (1,64)
Linear(64→2)                                  →  output (1,2)
```

Summaries are saved as UTF-8 text files to `Classifier Results/Visualization/`.

### 11.2 ONNX Export

ONNX (Open Neural Network Exchange) is a standardized format for representing ML models as a computation graph — nodes are operations (conv, matmul, relu), edges are tensors flowing between them. Exporting to ONNX lets you open the model in [Netron](https://netron.app), a browser-based graph viewer that renders the full architecture visually with clickable layers showing weights, kernel sizes, and tensor shapes.

Both models are exported with a dummy forward pass:
- `cnn1d.onnx` — dummy input `(1, 1, 4410)` representing one waveform
- `cnn_mfcc_head.onnx` — dummy input `(1, 1280)` representing one EfficientNet-B0 feature vector

> Note: The CNN_MFCC ONNX only covers the MLP head, not the EfficientNet-B0 backbone (which was not saved to `.pt`). To visualize the full EfficientNet architecture, export it separately before stripping the classifier.

### 11.3 Grad-CAM for CNN_1D

**Gradient-weighted Class Activation Mapping (Grad-CAM)** answers: *which parts of the input did the model focus on when predicting a given class?*

#### How it works

1. **Forward pass** — run a waveform `x` through the full network and obtain logits.

2. **Target the final conv layer** — we hook the last ReLU in `conv_blocks` (after the 4th Conv1d block). At this point the representation has shape `(B, 256, ~134)` — 256 channels, each covering a compressed version of the temporal axis.

3. **Backward pass** — compute the gradient of the target class score (e.g. Cracked confidence) with respect to those 256 feature maps:
   ```
   gradients = d(score_cracked) / d(activations)   shape: (256, ~134)
   ```
   A large gradient at position `(c, t)` means: "nudging channel `c` at time step `t` strongly changes the Cracked score."

4. **Weight the activation maps** — global-average-pool the gradients over the time axis to get one scalar weight per channel:
   ```
   alpha_c = mean_t( gradients[c, t] )   shape: (256,)
   ```
   Then compute the weighted sum of the activation maps:
   ```
   CAM = ReLU( sum_c( alpha_c * activations[c, :] ) )   shape: (~134,)
   ```
   ReLU discards negative contributions (regions that suppressed the class score).

5. **Upsample** — interpolate the `~134`-length CAM back to the original 4,410-sample input length so it aligns with the waveform. Normalize to `[0, 1]`.

#### Reading the plots

Each Grad-CAM plot has two panels:

- **Top panel:** the raw waveform colored by CAM intensity. Warm colors (red/yellow) = the network paid close attention here; cool colors (blue) = largely ignored.
- **Bottom panel:** the CAM signal as a filled area — a direct view of importance vs. time in milliseconds.

#### What to look for

- **Consistency across samples:** if all 3 Cracked samples light up the same region (e.g., 0–5 ms), the model learned a consistent acoustic discriminator. Scattered attention = overfitting or noise sensitivity.
- **Physical plausibility:** the initial transient (0–10 ms) encodes the impact and crack-modified stress wave reflection. Resonance decay (10–200 ms) encodes structural stiffness. Model attention in these regions would be physically meaningful.
- **Cracked vs Intact differences:** comparing the two classes' CAM maps shows what acoustic signature the model uses to separate them.

---

## 12. Level 2a — Crack Width Classification

### 12.1 Dataset

The 1,800 cracked WAV files split into 9 folders by specimen and measurement position:

| Folder | Width class | Position | Count |
|--------|-------------|----------|-------|
| specimen_1_1SoundDataC02 | 0.2 mm | 2 cm from crack | 200 |
| specimen_1_1SoundDataC04 | 0.2 mm | 4 cm from crack | 200 |
| specimen_1_2SoundDataC02 | 0.4 mm | 2 cm from crack | 200 |
| specimen_1_2SoundDataC04 | 0.4 mm | 4 cm from crack | 200 |
| specimen_1_3SoundDataC02 | 0.6 mm | 2 cm from crack | 200 |
| specimen_1_3SoundDataC04 | 0.6 mm | 4 cm from crack | 200 |
| specimen_2_1SoundDataC02 | 0.2 mm | 2 cm from crack | 200 |
| specimen_2_2SoundDataC02 | 0.4 mm | 2 cm from crack | 200 |
| specimen_2_3SoundDataC02 | 0.6 mm | 2 cm from crack | 200 |

**Classes are perfectly balanced** — 600 samples per width class, no weighting needed.

Labels are derived from the folder name: `specimen_X_Y` where `Y` encodes width (1→0.2mm label 0, 2→0.4mm label 1, 3→0.6mm label 2). Series 1 (`X=1`) has two positions (C02, C04); Series 2 (`X=2`) has only C02.

### 12.2 The Data Leakage Problem

Each physical specimen/position contributes exactly 200 nearly-identical hammer-strike recordings — the same concrete block, the same microphone position, the same acoustic environment, recorded 200 times. The only variation between recordings within a folder is minor hit-force fluctuations and microphone noise.

**With a random 80/20 split**, recordings from the same specimen appear in both train and test. For a given folder, roughly 160 recordings land in train and 40 in test. The model can achieve high accuracy simply by memorizing the acoustic *fingerprint* of each specimen — the characteristic frequency response of that particular block of concrete — rather than learning a generalizable relationship between crack width and acoustic properties.

This is classic **data leakage via specimen identity**: the model answers "which specimen does this sound like?" rather than "how wide is this crack?". Cross-validation within the same specimen pool is equally deceived — all folds contain near-duplicate recordings of the same physical specimens.

The symptom: ~99-100% accuracy with random splits, but the model has learned nothing meaningful about crack width physics.

### 12.3 Specimen-Level Split (Correct Methodology)

We split by physical specimen series:

- **Train:** Series 1 (`specimen_1_X` — both C02 and C04 positions per width class, 400 samples/class, 1200 total)
- **Test:** Series 2 (`specimen_2_X` — C02 only, 200 samples/class, 600 total)

This guarantees that no recording from a test specimen ever appears in training. The model must generalize to a physically different concrete specimen it has never seen — which is the actual real-world task.

Label extraction:
```python
width_digit = df["Specimen"].str.extract(r"specimen_\d+_(\d+)")[0].astype(int)
series      = df["Specimen"].str.extract(r"specimen_(\d+)_\d+")[0].astype(int)
y       = (width_digit - 1).values   # 1->0, 2->1, 3->2
is_test = (series == 2).values       # True -> test set
```

For the MLP, a random 10% split within series 1 is used for the validation set during training (early stopping). This small leakage is acceptable since the test boundary is already clean — the validation contamination is between C02 and C04 recordings of the same series-1 specimens, not between series.

### 12.4 Results

| Model | CV Acc (series 1) | CV F1 (series 1) | Test Acc (series 2) | Test Macro F1 (series 2) |
|-------|-------------------|------------------|---------------------|--------------------------|
| Random Forest | 99.75% | 99.75% | 34.5% | 35.1% |
| XGBoost | 99.67% | 99.67% | 41.8% | 38.2% |
| SVM | 99.83% | 99.83% | 33.5% | 17.0% |
| MLP | ~99% (val F1=1.00) | — | 33.2% | 19.2% |

### 12.5 Interpretation

The gap between CV accuracy (~99%) and test accuracy (33–42%) is the clearest possible evidence of overfitting to specimen identity. A 3-class random baseline would score 33.3% — meaning the SVM and MLP, with test F1 of 17–19%, perform worse than random. They have confidently learned the wrong thing.

**Why this is hard:** The dataset contains only 2 distinct physical specimens per width class (series 1 and series 2). With specimen-level splitting, each model trains on 1 specimen and tests on 1. A single training specimen is an insufficient basis for learning generalizable acoustic width features — especially when the two series differ not just in crack width but also in crack depth and specimen geometry (series 1 cracks are 40mm deep, series 2 are 20mm deep). These correlated differences make it impossible to isolate width as a clean signal.

**Comparison to the paper:** The paper's reported 99.44% SVM accuracy for width classification almost certainly reflects a random-sample split across the same set of specimen folders, subject to the same leakage. The paper does not describe a specimen-level evaluation.

**Key takeaway:** Crack width classification from acoustic features requires more physical specimens to establish genuine cross-specimen generalizability. With the current dataset (2 specimens per class), any within-dataset random split evaluation is unreliable as a measure of real-world performance.
