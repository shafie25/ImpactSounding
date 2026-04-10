# Dataset Notes: Structure, Preparation, and Limitations

---

## 1. How the Data Points Were Created

### Step 1 — Physical Specimen Fabrication

Six concrete blocks were cast in steel molds (400 mm × 100 mm × 100 mm). Before casting, a polyethylene sheet was embedded inside each block at a 45° angle to simulate a diagonal shear crack. After 28 days of curing, the sheet was pulled out, leaving an artificial crack.

The 6 specimens split into two groups based on crack length:

| Series | Specimens | Crack Length (diagonal) | Vertical Depth | Crack Widths |
|--------|-----------|------------------------|----------------|--------------|
| 1 | 1.1, 1.2, 1.3 | 80 mm | ~40 mm | 0.2 mm, 0.4 mm, 0.6 mm |
| 2 | 2.1, 2.2, 2.3 | 40 mm | ~20 mm | 0.2 mm, 0.4 mm, 0.6 mm |

Within each series, each specimen has one crack width (the trailing number in the specimen name — `_1` = 0.2 mm, `_2` = 0.4 mm, `_3` = 0.6 mm).

### Step 2 — Hammering Line Setup

On the top surface of each block, measurement lines were drawn perpendicular to where the crack meets the surface. These lines were spaced at 2 cm intervals, labeled by their distance from the crack and which side they fall on:

- **C** = Cracked side (the hammer strike passes over the crack region)
- **N** = Intact side (away from the crack)
- **Number** = distance in cm from the crack (02, 04, 06, 08, 10)

So `C02` = cracked side, 2 cm from the crack. `N04` = intact side, 4 cm from the crack.

Series 1 specimens have positions: C02, C04, C06, C08, C10, N02, N04, N06, N08, N10  
Series 2 specimens have positions: C02, C04, C06, C08, C10, N02, N04, N06, N08, N10  
**However, only C02 and C04 are labeled Cracked.** C06 and beyond are labeled Intact even on the cracked side (too far from the crack to be affected). Series 2 cracked data only contains C02 — C04 was not collected for series 2.

### Step 3 — Recording the Hammer Hits

A controlled recording environment was used: the specimen was placed inside a cardboard box lined with high-density foam for acoustic isolation. A Sony condenser microphone was hung inside the box. An inspector continuously struck along each measurement line with a double-ended impact hammer while the microphone recorded the full session as a single continuous audio file.

### Step 4 — Segmentation into Individual Hits

The continuous recording for each line was processed by a segmentation algorithm that detected the onset and offset of each individual hammer impact and cut it into a separate WAV file. Each WAV file is one hammer hit — approximately 0.20 seconds long at 22,050 Hz (4,410 samples).

**This is where the individual data points come from.** Each WAV file = one row in the dataset.

### Step 5 — Feature Extraction

For each WAV file, three types of features were extracted using Python and librosa:

- **Df** (Dominant Frequency): the frequency with the highest amplitude in the FFT spectrum.
- **Vf** (Frequency Feature Value): an energy-weighted spread of the FFT spectrum — a single scalar describing how distributed the frequency content is.
- **MFCCs**: 13 Mel-frequency cepstral coefficients, each summarized by its mean and standard deviation across time frames → 26 values.

Together with Df and its amplitude (2 values) and Vf (1 value), this gives **29 features** per hit, stored in `Labeled_Features.xlsx`.

### Step 6 — Labeling

Each row was labeled based on which folder its WAV file came from:

| Task | Label | Assigned to |
|------|-------|-------------|
| Crack Detection | Cracked | Any C02 or C04 position |
| Crack Detection | Intact | C06, C08, C10, all N positions |
| Crack Width | 0.2 mm | Any specimen ending in `_1` |
| Crack Width | 0.4 mm | Any specimen ending in `_2` |
| Crack Width | 0.6 mm | Any specimen ending in `_3` |
| Crack Depth | 20 mm | Any Cracked row at C02 |
| Crack Depth | 40 mm | Any Cracked row at C04 (series 1 only) |

---

## 2. Final Dataset Counts

| Class | Count |
|-------|-------|
| **Intact** (all positions, both series) | 8,236 |
| **Cracked** (C02 + C04, both series) | 1,800 |
| **Total** | 10,036 |

Cracked breakdown by task:

| Width Class | Count | Depth Class | Count |
|------------|-------|-------------|-------|
| 0.2 mm | 600 | 20 mm (C02) | 1,200 |
| 0.4 mm | 600 | 40 mm (C04) | 600 |
| 0.6 mm | 600 | | |

---

## 3. Problems This Dataset Structure Creates

### Problem 1 — Specimen-Level Data Leakage (Crack Width)

**What the paper did:** Random 70/30 split across all 1,800 cracked samples. Result: **99.44% accuracy**.

**The issue:** Each physical block produced ~200 hammer hits (one row per hit). A random split puts hits from the *same physical block* into both training and testing. The model does not need to learn what a 0.2 mm crack sounds like in general — it only needs to recognize the acoustic fingerprint of a specific block it has already seen during training.

Every block has a unique resonance profile shaped by its exact geometry, material variation, and surface texture. These differences between blocks are much larger than the differences between crack widths. When the test set contains hits from the same block as the training set, the model trivially memorizes the block identity rather than learning crack width physics.

**What happens with a proper split:** When we train on series 1 specimens and test on series 2 specimens (completely unseen physical blocks), accuracy drops to **~34–42%** — essentially random guessing for a 3-class problem (33% baseline).

**Why this can't be fixed with this dataset:** There are only 2 physical blocks per width class (one per series). You cannot learn a generalizable width signal from 1 training example per class at the specimen level.

**Does the paper acknowledge this?** No. The paper reports 99.44% without discussing specimen-level leakage.

---

### Problem 2 — Specimen-Level Data Leakage (Crack Depth)

The same leakage problem exists for depth classification, but it is more nuanced.

**What the paper did:** Random 70/30 split across all 1,200 cracked samples used for depth. Result: **96.67% accuracy**.

**What happens with a proper split:** We held out specimen `_3` of both series as the test set (the closest we can get to an unseen-specimen evaluation). Results with the specimen-level split:

| Model | Test Accuracy |
|-------|--------------|
| Random Forest | 71.7% |
| XGBoost | 78.3% |
| SVM | 80.8% |
| MLP | 89.3% |
| 1D CNN | 65.7% |

These numbers are higher than the width collapse, but this does not necessarily mean the model is learning crack depth. See Problem 3.

---

### Problem 3 — Position Confound in Depth Classification

The depth labeling is inseparable from the hammering position:

- **20 mm depth** → always recorded at **C02** (2 cm from crack)
- **40 mm depth** → always recorded at **C04** (4 cm from crack)

This means the depth classifier is simultaneously trained on two things:
1. The difference in crack depth (20 mm vs 40 mm)
2. The difference in strike distance from the crack (2 cm vs 4 cm)

It is physically impossible to know from this dataset whether the model learned "depth" or "distance from crack." The two signals are perfectly correlated in the data. A strike at C02 sounds different from one at C04 for purely geometric reasons (the wave has less concrete to travel through before hitting the crack at C02), and this geometric difference would generalize across specimens even if depth itself does not.

This is why depth results look better than width after a specimen-level split — the C02/C04 position signal is real and transferable, but it is not the same thing as learning crack depth.

**The only way to isolate the depth signal** would be to record all depth cases at the same position (e.g., C02 only for both 20 mm and 40 mm specimens), which was not done in this study.

---

### Problem 4 — Depth Classification Has No Cross-Series Test for the 40 mm Class

For crack width, we could do a clean series-1-train / series-2-test split because both series had all three width classes.

For crack depth, **series 2 specimens only have C02 data**. There is no C04 data for series 2, meaning:
- The 40 mm class only exists in series 1.
- You can never evaluate the 40 mm class on a specimen that was not seen during training (if using a series-level split).

The best possible split for depth is leave-one-specimen-out within series 1, which is what we implemented. But this still only withholds one physical block per class from the training set.

---

## 4. Summary of Limitations

| Task | Paper Accuracy | Our Proper Split | Root Cause |
|------|---------------|-----------------|------------|
| Crack Detection | 97.59% | 99.70% (our SVM) | No leakage issue here — intact vs. cracked is a strong, generalizable signal |
| Crack Width | 99.44% | ~34–42% | Specimen-level leakage; only 2 blocks per class; no generalizable width signal found |
| Crack Depth | 96.67% | 65–89% | Specimen-level leakage + position confound; apparent signal may be C02 vs C04 distance, not true depth |

The crack detection task is reliable. The crack width and depth results from the paper are not reproducible under a proper evaluation methodology, and the dataset is too small (6 physical specimens total) to resolve this.
