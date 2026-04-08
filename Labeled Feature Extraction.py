# ============================================================
# Labeled Feature Extraction
# Acoustic Impact Hammer Testing
#
# This script:
# 1) Walks Specimens/Class_sounds/Cracked/ and Intact/
# 2) Extracts Df, Vf, and MFCC statistical features
#    for every WAV file.
# 3) Attaches Label (Cracked / Intact) and
#    Crack_Position (C02 / C04 / N/A) to each row.
# 4) Saves one labeled feature table ready for ML.
# ============================================================

import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa


# ============================================================
# PARAMETERS
# ============================================================

CLASS_SOUNDS_DIR = os.path.join("Specimens", "Class_sounds")
OUTPUT_FILE      = "Labeled_Features.xlsx"
NUM_MFCC         = 13


# ============================================================
# HELPER — extract all features from one WAV file
# ============================================================

def extract_features(audio_path):

    signal, sr = librosa.load(audio_path, sr=None, mono=True)

    # ---- Dominant Frequency (Df) via FFT ----
    fft_mag   = np.abs(np.fft.rfft(signal))
    freq_axis = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    peak_idx  = np.argmax(fft_mag)
    df_hz     = float(freq_axis[peak_idx])
    df_amp    = float(fft_mag[peak_idx])

    # ---- Frequency Feature Value (Vf) ----
    amp_sq   = fft_mag ** 2
    as_sq    = np.sum(amp_sq)
    if as_sq == 0:
        vf = 0.0
    else:
        as_val = np.sqrt(as_sq)
        f_bar  = np.sum(amp_sq * freq_axis) / as_sq
        vf     = float(np.sqrt(np.sum(amp_sq * (freq_axis - f_bar) ** 2)) / as_val)

    # ---- MFCC statistics (mean + std per coefficient) ----
    mfcc_matrix = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
    mfcc_means  = mfcc_matrix.mean(axis=1)
    mfcc_stds   = mfcc_matrix.std(axis=1)

    features = {
        "Df_Hz":        df_hz,
        "Df_Amplitude": df_amp,
        "Vf":           vf,
    }
    for i in range(NUM_MFCC):
        features[f"MFCC_{i+1}_mean"] = float(mfcc_means[i])
        features[f"MFCC_{i+1}_std"]  = float(mfcc_stds[i])

    return features


# ============================================================
# MAIN LOOP — walk Cracked and Intact folders
# ============================================================

rows = []
total_files = sum(
    len(glob.glob(os.path.join(CLASS_SOUNDS_DIR, cls, "**", "*.wav"), recursive=True))
    for cls in ["Cracked", "Intact"]
)
processed = 0

for label in ["Cracked", "Intact"]:

    label_dir = os.path.join(CLASS_SOUNDS_DIR, label)

    for wav_path in glob.glob(os.path.join(label_dir, "**", "*.wav"), recursive=True):

        # ---- Build a readable Signal_ID ----
        # e.g. specimen_1_1SoundDataC02/003
        rel_path  = os.path.relpath(wav_path, label_dir)
        signal_id = os.path.splitext(rel_path)[0].replace("\\", "/")

        # ---- Extract crack position from subfolder name ----
        # Matches C02, C04, etc. — present only in Cracked subfolders
        subfolder     = rel_path.split(os.sep)[0]
        position_match = re.search(r"C\d+", subfolder)
        crack_position = position_match.group() if position_match else "N/A"

        # ---- Extract features ----
        try:
            features = extract_features(wav_path)
        except Exception as e:
            print(f"  Skipping {signal_id}: {e}")
            continue

        row = {
            "Signal_ID":      signal_id,
            "Label":          label,
            "Crack_Position": crack_position,
            "Specimen":       subfolder,
            **features
        }
        rows.append(row)

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{total_files} files...")


# ============================================================
# Save to Excel
# ============================================================

print(f"\nDone. {len(rows)} files extracted.")

df = pd.DataFrame(rows)

# Reorder columns for readability
meta_cols    = ["Signal_ID", "Label", "Crack_Position", "Specimen"]
feature_cols = [c for c in df.columns if c not in meta_cols]
df = df[meta_cols + feature_cols]

print(f"\nClass distribution:")
print(df["Label"].value_counts().to_string())
print(f"\nCrack positions:")
print(df["Crack_Position"].value_counts().to_string())

df.to_excel(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
