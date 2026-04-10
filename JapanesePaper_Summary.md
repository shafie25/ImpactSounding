# Paper Summary: Artificial Intelligence Enhanced Automatic Identification for Concrete Cracks Using Acoustic Impact Hammer Testing

**Authors:** Mohamad Najib Alhebrawi, Huang Huang, Zhishen Wu  
**Institutions:** Ibaraki University (Japan), Southeast University (China)  
**Published:** Journal of Civil Structural Health Monitoring, 2023, Vol. 13, pp. 469–484  
**DOI:** 10.1007/s13349-022-00651-8

---

## 1. Problem Statement and Motivation

Concrete crack detection in reinforced concrete (RC) structures is critical for maintenance planning. Impact hammer testing (tapping a surface and listening to the sound) is a standard inspection method used on bridges, tunnels, and other structures. The problem is that:

- Manual inspection depends on the experience and skill of the inspector.
- Unskilled inspectors arrive at incorrect conclusions.
- Prior methods using dominant frequency (Df) from FFT worked for large defects (>100 mm wide cavities/voids) but failed for fine cracks.
- No reliable automated system existed for detecting fine cracks (as narrow as 0.2 mm) by their depth or width.

The paper proposes a fully automated pipeline to identify:
1. Whether a location is cracked or intact (Level 1)
2. The width of the crack: 0.2 mm, 0.4 mm, or 0.6 mm (Level 2a)
3. The depth of the crack: 20 mm or 40 mm (Level 2b)

---

## 2. Acoustic Feature Theory

### 2.1 Why Sound Changes at a Crack

Solid concrete stores strain energy internally. When it is damaged, the way that energy transfers changes — this causes measurable shifts in the frequency characteristics of the sound produced when the surface is struck.

The paper investigates three acoustic features derived from the FFT of the hammer impact sound:

### 2.2 Feature 1: Dominant Frequency (Df)

- Defined as the frequency at which the FFT power spectrum reaches its **maximum amplitude** (the peak frequency).
- Commonly used in prior work for detecting delamination and voids.
- Computed by finding the `argmax` of the amplitude in the FFT spectrum.
- Prior work showed it works for defects >100 mm wide. For fine cracks it was expected to be limited.

### 2.3 Feature 2: Frequency Feature Value (Vf)

- A single scalar that describes the **spread/distribution** of the FFT spectrum, not just the peak.
- Computed using the energy-weighted standard deviation of frequencies:

$$V_f = \frac{1}{A_s} \sqrt{\sum_{i=0}^{n-1} \left\{ (f_i - \bar{f})^2 \times A_i^2 \right\}}$$

where:

$$\bar{f} = \frac{\sum_{i=0}^{n-1} f_i A_i^2}{A_s^2}, \quad A_s^2 = \sum_{i=0}^{n-1} A_i^2$$

- $f_i$ = i-th frequency bin, $A_i$ = i-th amplitude, $n$ = number of FFT points.
- Advantage over Df: uses the full spectral shape, not just one point.

### 2.4 Feature 3: Mel-Frequency Cepstral Coefficients (MFCCs)

- The primary feature used in the AI approach.
- Originally developed for speech recognition; well-suited to non-stationary signals like hammer impacts.
- The **Mel scale** mimics the human ear's nonlinear frequency perception — it rises quickly at low frequencies and slowly at high frequencies:

$$m = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)$$

- MFCCs capture time-frequency information in 2D, reflecting non-steady-state characteristics of sound.
- More noise-robust than linear prediction coefficients.
- 13 MFCCs were computed per sound sample.
- The **first coefficient (MFCC #1) was dropped** because it is overly sensitive to overall signal energy and hurts model accuracy.
- Therefore 12 MFCCs were used, each represented by its **max** and **mean** value across time frames.
- MFCC computation parameters: chunk size = 512 samples, overlap = 25%.
- Software: KNIME platform using the JAudio feature extraction plugin.

---

## 3. Specimen Design and Fabrication

### 3.1 Concrete Mix

- Targeted compressive strength: 30 MPa after 28 days of curing.
- Actual average 28-day strength: ~29 MPa (verified using 200 mm × 100 mm cylindrical specimens on a 500 kN compression machine).

### 3.2 Specimen Dimensions

- 6 specimens total, cast in steel molds: **400 mm × 100 mm × 100 mm**.
- Divided into two groups (Series 1 and Series 2) with 3 specimens each.

### 3.3 Artificial Crack Creation

- Artificial cracks were created by embedding **polyethylene sheets** into the mold before casting and pulling them out after curing.
- Sheet thickness = 0.2 mm per layer, so:
  - 0.2 mm crack = 1 layer
  - 0.4 mm crack = 2 layers
  - 0.6 mm crack = 3 layers
- Both sides of the polyethylene were greased with heavy oil to ease removal.
- The mold itself was also greased for clean demolding after 28 days.
- Crack angle: **45°** (diagonal shear crack — the most common type in RC structures under service loads).

### 3.4 Crack Dimensions per Specimen

| Specimen | Width (mm) | Crack Length (mm) | Crack Depth (mm) |
|----------|------------|-------------------|------------------|
| 1.1      | 0.2        | 80                | 56.6             |
| 1.2      | 0.4        | 80                | 56.6             |
| 1.3      | 0.6        | 80                | 56.6             |
| 2.1      | 0.2        | 40                | 28.3             |
| 2.2      | 0.4        | 40                | 28.3             |
| 2.3      | 0.6        | 40                | 28.3             |

- Series 1 (specimens 1.1–1.3): 80 mm long crack → depth = 56.57 mm → vertical component ≈ **40 mm** (over the steel reinforcement position).
- Series 2 (specimens 2.1–2.3): 40 mm long crack → depth = 28.28 mm → vertical component ≈ **20 mm** (equal to concrete cover).
- The 45° angle means the crack length along the diagonal is longer than the vertical depth. The 20 mm and 40 mm figures refer to the **vertical depth** into the concrete.

### 3.5 Physical Motivation for These Dimensions

- ACI code specifies concrete cover ≥ 20 mm → chosen as the "shallow" crack depth case.
- ACI 224R-01 specifies maximum reasonable crack width under service loads = 0.41 mm → three cases bracket this: below (0.2 mm), near (0.4 mm), and above (0.6 mm).

---

## 4. Experimental Setup and Data Collection

### 4.1 Hammering Environment

- A controlled acoustic environment was used to ensure clean data with minimal background noise.
- A **cardboard box** lined on the inside with **500 mm × 400 mm × 40 mm high-density absorbent foam** served as an isolation shield.
- The microphone was hung inside the box.
- Equipment:
  - **Microphone:** Sony ECM-PCV80U electric condenser vocal microphone
  - **Hammer:** Double-ended impact hammer
  - **Laptop:** for recording

### 4.2 Hammering Lines and Positions

For each specimen, measurement lines were drawn **perpendicular to the crack**, 5 cm long, starting 2 cm on either side of the crack surface location. These lines simulate a real inspector walking along the surface and striking it.

Line labels:
- **C** = Cracked side (hammer strike passes over the crack)
- **N** = Non-cracked (intact) side

Number suffixes (02, 04, 06, 08, 10) indicate the **perpendicular distance in cm** from the crack to the hammering line:
- C02 = cracked side, 2 cm from crack
- C04 = cracked side, 4 cm from crack
- C06, C08, C10 = cracked side, 6/8/10 cm from crack (increasingly far, transitions to intact behavior)
- N02, N04, N06, N08, N10 = intact side, 2–10 cm from crack

### 4.3 Data Labeling for Each Task

| Task | Class | Lines Used |
|------|-------|-----------|
| Crack detection | Intact | Series 1: C06, C08, C10 + all N lines; Series 2: C04, C06, C08, C10 + all N lines |
| Crack detection | Cracked | Series 1: C02, C04; Series 2: C02 only |
| Crack depth | 20 mm depth | Series 2 specimens (2.1, 2.2, 2.3) — C02 |
| Crack depth | 40 mm depth | Series 1 specimens (1.1, 1.2, 1.3) — C04 |
| Crack width | 0.2 mm | Specimen 1.1 (C02+C04) + Specimen 2.1 (C02) |
| Crack width | 0.4 mm | Specimen 1.2 (C02+C04) + Specimen 2.2 (C02) |
| Crack width | 0.6 mm | Specimen 1.3 (C02+C04) + Specimen 2.3 (C02) |

### 4.4 Data Collection Volume

- 200 hammer hits per line per specimen were recorded.
- Each hammer hit was recorded as a continuous audio stream, then **segmented** into individual hit samples in preprocessing.

---

## 5. Analytical Analysis (Non-AI Methods)

Before applying AI, the paper analyzed Df and Vf manually to see if they alone could detect or classify cracks.

### 5.1 Df Results

- 100 random hammer hits per case were analyzed.
- Averaged Df values were plotted for each crack case.
- **Finding:** Df is higher in cracked locations than intact — it can distinguish cracked from intact.
- **Limitation:** No consistent relationship between Df and crack width or depth. For example:
  - At 20 mm depth: the 0.6 mm crack had the highest Df, but 0.4 mm was near intact, and 0.2 mm was in the middle — no monotonic trend.
  - At 40 mm depth: all crack Df values dropped toward the intact value, making deep cracks hard to detect.
- **Root cause:** The defects in this study were extremely narrow (0.2–0.6 mm) vs. prior studies where Df worked on defects >100 mm wide.

### 5.2 Vf Results

- Vf was computed from the same FFT data for the same 100 hits per case.
- **Finding:** Intact locations had the lowest Vf; shallow (20 mm) cracks had a higher Vf — detectable.
- **Limitation 1:** Could not distinguish between 0.2 mm and 0.4 mm crack widths.
- **Limitation 2:** For deep cracks (40 mm), Vf values fell back toward the intact level, making deep cracks invisible to this feature.
- **Root cause:** Same as Df — the crack widths studied are too fine for simple FFT-based scalar features.

### 5.3 Conclusion from Analytical Methods

Both Df and Vf:
- Can loosely indicate cracked vs. intact (higher value = cracked).
- Cannot reliably classify crack depth or crack width.
- Therefore, a richer feature (MFCC) and machine learning are required.

---

## 6. AI-Enhanced Approach

### 6.1 Feature Extraction Details

- **Tool:** KNIME platform with JAudio plugin.
- **Parameters:** 13 MFCCs, chunk size = 512 samples, 25% overlap.
- **Per-coefficient descriptors:** Max value and Mean value across time frames.
- **MFCC #1 dropped** (too sensitive to DC energy, hurts accuracy).
- **Final feature vector:** 12 coefficients × 2 statistics (max + mean) = **24 features** per hammer hit.
- Features normalized to a common scale using a normalizer node before training.

### 6.2 Dataset Balancing

- **Equal Size Sampling (ESS)** was applied to ensure all classes have the same number of samples before training.
- This prevents models from being biased toward the majority class.

### 6.3 Train/Test Split

- **70% training, 30% testing** for all three models (FR, GBT, SVM).
- Splits were **random** — no specimen-level isolation was applied.

### 6.4 Three AI Algorithms

#### Algorithm 1: Fuzzy Rule (FR)

- Based on fuzzy logic — generates interpretable if-then rules directly from data.
- Uses "mixed fuzzy rule formation" as the training algorithm.
- Weakest performer in all three tasks.

#### Algorithm 2: Gradient Boosted Trees (GBT)

- Ensemble method: trains many weak decision trees sequentially, each correcting the errors of the previous.
- Strong middle performer — consistently above 93% in all tasks.
- No hyperparameter tuning details given for GBT specifically.

#### Algorithm 3: Support Vector Machine (SVM) — Primary Model

- Uses a **Radial Basis Function (RBF) kernel**.
- Two hyperparameters tuned:
  - **C** (overlapping penalty): searched in range [0.1, 10.0]
  - **γ (gamma)**: searched in range [0.1, 5.0]
- Optimization method: **Bayesian optimizer using Tree-structured Parzen Estimation (TPE)**, run for 100 iterations.
- Each iteration: 5-fold cross-validation (80/20 split ratio within the CV).
- Best hyperparameters selected = those yielding maximum cross-validation accuracy.
- Final model trained on 70% of data with best hyperparameters, tested on remaining 30%.
- All three models were trained and evaluated on the **same dataset** for fair comparison.

---

## 7. Results

### 7.1 Level 1: Crack Detection (Cracked vs. Intact)

- **Total dataset (after ESS):** 3,600 hits
- **Training set:** 2,520 hits | **Test set:** 1,080 hits
- **Best SVM hyperparameters:** C = 5.33, γ = 0.37

| Model | Accuracy |
|-------|----------|
| SVM   | **97.59%** |
| GBT   | 93.98% |
| FR    | 90.84% |

**SVM Detailed Statistics (Confusion Matrix):**

|               | Predicted Cracked | Predicted Intact |
|---------------|-------------------|------------------|
| Actual Cracked | 525 (TP)          | 15 (FN)          |
| Actual Intact  | 11 (FP)           | 529 (TN)         |

| Metric | Cracked | Intact |
|--------|---------|--------|
| Recall (Sensitivity) | 97.22% | 97.96% |
| Precision | 97.95% | 97.24% |
| F-measure | 97.58% | 97.60% |
| Specificity | 97.96% | 97.22% |

- Overall accuracy: **97.59%**, error: 2.41%, Cohen's κ = 0.952
- Correctly classified: 1,054 / 1,080 | Incorrectly classified: 26

---

### 7.2 Level 2a: Crack Width Identification (0.2 / 0.4 / 0.6 mm)

- **Total dataset (after ESS):** 1,800 hits
- **Training set:** 1,260 hits | **Test set:** 540 hits
- **Best SVM hyperparameters:** C = 7.04, γ = 0.80

| Model | Accuracy |
|-------|----------|
| SVM   | **99.44%** |
| GBT   | 98.52% |
| FR    | 94.32% |

**SVM Detailed Statistics (Confusion Matrix):**

|                      | Pred 0.2 mm | Pred 0.4 mm | Pred 0.6 mm |
|----------------------|-------------|-------------|-------------|
| Actual 0.2 mm        | 180         | 0           | 0           |
| Actual 0.4 mm        | 1           | 178         | 1           |
| Actual 0.6 mm        | 0           | 1           | 179         |

| Class | Recall | Precision | F-measure |
|-------|--------|-----------|-----------|
| 0.2 mm | 100.00% | 99.45% | 99.72% |
| 0.4 mm | 98.89% | 99.44% | 99.16% |
| 0.6 mm | 99.44% | 99.44% | 99.44% |

- Overall accuracy: **99.44%**, error: 0.56%, Cohen's κ = 0.992
- Correctly classified: 537 / 540 | Incorrectly classified: 3
- **Note:** This is the highest accuracy of the three tasks.

---

### 7.3 Level 2b: Crack Depth Identification (20 mm vs. 40 mm)

- **Total dataset (after ESS):** 1,200 hits
- **Training set:** 840 hits | **Test set:** 360 hits
- **Best SVM hyperparameters:** C = 7.53, γ = 0.91

| Model | Accuracy |
|-------|----------|
| SVM   | **96.67%** |
| GBT   | 94.17% |
| FR    | 90.12% |

**SVM Detailed Statistics (Confusion Matrix):**

|                     | Pred 20 mm | Pred 40 mm |
|---------------------|------------|------------|
| Actual 20 mm        | 173        | 7          |
| Actual 40 mm        | 5          | 175        |

| Class | Recall | Precision | F-measure |
|-------|--------|-----------|-----------|
| 20 mm depth | 96.11% | 97.19% | 96.65% |
| 40 mm depth | 97.22% | 96.15% | 96.69% |

- Overall accuracy: **96.67%**, error: 3.33%, Cohen's κ = 0.933
- Correctly classified: 348 / 360 | Incorrectly classified: 12

---

### 7.4 Summary of All Results

| Task | Best Model | Accuracy | Cohen's κ |
|------|-----------|----------|-----------|
| Crack Detection (Cracked vs. Intact) | SVM | 97.59% | 0.952 |
| Crack Width (0.2/0.4/0.6 mm) | SVM | 99.44% | 0.992 |
| Crack Depth (20/40 mm) | SVM | 96.67% | 0.933 |

- SVM outperformed GBT and FR in all three tasks.
- GBT was consistently second best (>93% in all tasks).
- FR was consistently weakest (90–94%).
- Crack width was the **easiest task** for the model — the paper suggests each crack width produces a distinctly different MFCC fingerprint.
- Crack depth was the **hardest task**.

---

## 8. Key Design Choices and Justifications

| Decision | Choice | Reason |
|----------|--------|--------|
| Number of MFCCs | 13 (then drop #1 → use 12) | Standard for speech recognition; #1 dropped due to over-sensitivity to DC energy |
| MFCC descriptors | Max + Mean per coefficient | Captures peak and average behavior across time |
| Chunk size | 512 samples | Consistent with prior MFCC hammering studies |
| Overlap | 25% | Consistent with prior work |
| Train/test split | 70/30 | Shown in prior literature to give best model performance |
| Class balancing | Equal Size Sampling (ESS) | Ensures no class dominates training |
| SVM kernel | RBF | Standard for nonlinear classification |
| Hyperparameter search | Bayesian TPE, 100 iterations | More efficient than grid search |
| CV folds | 5-fold | Standard; 80/20 ratio within each fold |
| Platform | KNIME (no-code workflow) | Used for all data processing, training, and evaluation |

---

## 9. Limitations Acknowledged by the Paper

1. **Environmental noise not studied** — vehicle noise, construction site noise were not tested. The experiment was conducted in a controlled anechoic box.
2. **Only diagonal (45°) shear cracks studied** — flexural (vertical) cracks were not included.
3. **Only one crack per specimen** — double cracks or multi-level cracks were not tested.
4. **Random train/test split** — specimens from the same physical block appear in both train and test sets (data leakage risk not discussed by the authors).
5. **Small dataset** — only 6 physical specimens total.

---

## 10. Proposed Future Work

- Combine the AI approach with a **robotic hammering system** for autonomous real-time inspection.
- Add a **noise-cancellation algorithm** so the method works in noisy field environments.
- Apply to areas difficult to access: bridges, tunnels.
- Study vertical (flexural) cracks and multiple simultaneous cracks.
- Expand to other concrete defect types.

---

## 11. Software and Tools Used

| Tool | Purpose |
|------|---------|
| Python + librosa | FFT extraction for Df analysis |
| PyCharm | Python IDE |
| KNIME platform | Full AI pipeline (feature extraction, normalization, ESS, training, evaluation) |
| JAudio (KNIME plugin) | MFCC feature extraction |

---

## 12. Our Reproduction Notes

Our reimplementation in Python/sklearn/PyTorch reproduces the paper's pipeline with the following differences:

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| MFCC descriptors | Max + Mean, 12 coefficients | Mean + Std, 13 coefficients |
| Feature extraction tool | KNIME JAudio | Python librosa |
| Classifiers | FR, GBT, SVM | SVM, XGBoost, RF, MLP, 1D CNN, LSTM, EfficientNet-CNN |
| Train/test split | 70/30 random | 80/20 random (Level 1) + specimen-level split (Level 2a) |
| Platform | KNIME (no-code) | Python scripts |
| Crack detection accuracy (SVM) | 97.59% | **99.70%** (our SVM) |
| Crack width accuracy (SVM) | 99.44% (random split) | ~99% (random split) / **~35–42%** (specimen-level split — reveals data leakage) |
