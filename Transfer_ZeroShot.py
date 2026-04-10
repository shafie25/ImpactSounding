# ============================================================
# Transfer_ZeroShot.py
# Zero-Shot Transfer Learning — Bridge Dataset (JapanDataset2)
#
# Tests whether our crack detection models trained on lab
# concrete specimens generalize to real bridge hammer recordings.
#
# Pipeline per bridge file:
#   1. Load WAV, resample 44100 Hz -> 22050 Hz
#   2. Detect individual hammer impacts via energy peaks
#   3. Extract 0.20s (4410-sample) window per impact
#   4. Extract same features as lab pipeline (Df, Vf, 13 MFCCs)
#   5. Run through all trained models (no retraining)
#   6. Majority vote across impacts -> one prediction per file
#
# Labels:  Normal = Intact (1),  Abnormal = Cracked (0)
# (matches LabelEncoder alphabetical order from lab training)
#
# Models loaded from disk (no retraining):
#   - Random Forest  -> Classifier Results/CrackDetection/RandomForest/model.joblib
#   - XGBoost        -> Classifier Results/CrackDetection/XGBoost/model.joblib
#   - SVM            -> Classifier Results/CrackDetection/SVM/model.joblib
#   - MLP            -> Classifier Results/CrackDetection/MLP/best_model.pt
#                       + scaler.joblib for feature normalisation
#   - 1D CNN         -> Classifier Results/CrackDetection/CNN_1D/best_model.pt
#
# Output: Classifier Results/TransferLearning/
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import find_peaks

import librosa
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn


# ============================================================
# PARAMETERS
# ============================================================

BRIDGE_DIR      = "JapanDataset2"
BRIDGE_LABELS   = "JapanDataset2Labels.csv"
OUTPUT_DIR      = os.path.join("Classifier Results", "TransferLearning")

RF_MODEL        = os.path.join("Classifier Results", "CrackDetection", "RandomForest", "model.joblib")
XGB_MODEL       = os.path.join("Classifier Results", "CrackDetection", "XGBoost",      "model.joblib")
SVM_MODEL       = os.path.join("Classifier Results", "CrackDetection", "SVM",          "model.joblib")
MLP_WEIGHTS     = os.path.join("Classifier Results", "CrackDetection", "MLP", "best_model.pt")
MLP_SCALER      = os.path.join("Classifier Results", "CrackDetection", "MLP", "scaler.joblib")
CNN_WEIGHTS     = os.path.join("Classifier Results", "CrackDetection", "CNN_1D", "best_model.pt")

RANDOM_STATE    = 42
TARGET_SR       = 22050
NUM_SAMPLES     = 4410          # 0.20s at 22050 Hz
ONSET_OFFSET    = 220           # 10ms before peak (samples at 22050 Hz)

# Minimum gap between impacts: 0.25s in energy frames (hop=256)
MIN_PEAK_DIST   = int(0.25 * TARGET_SR / 256)

FEATURE_COLS = (
    ["Df_Hz", "Df_Amplitude", "Vf"] +
    [col for i in range(1, 14) for col in (f"MFCC_{i}_mean", f"MFCC_{i}_std")]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION  (identical to lab pipeline)
# ============================================================

def extract_features(audio, sr):
    """Extract Df, Vf, 13 MFCC mean+std from a waveform segment."""
    # Dominant Frequency
    fft_mag   = np.abs(np.fft.rfft(audio))
    freq_axis = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    peak_idx  = np.argmax(fft_mag)
    df_hz     = float(freq_axis[peak_idx])
    df_amp    = float(fft_mag[peak_idx])

    # Frequency Feature Value (Vf)
    amp_sq = fft_mag ** 2
    as_sq  = np.sum(amp_sq)
    if as_sq == 0:
        vf = 0.0
    else:
        as_val = np.sqrt(as_sq)
        f_bar  = np.sum(amp_sq * freq_axis) / as_sq
        vf     = float(np.sqrt(np.sum(amp_sq * (freq_axis - f_bar) ** 2)) / as_val)

    # MFCCs
    mfcc_matrix = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means  = mfcc_matrix.mean(axis=1)
    mfcc_stds   = mfcc_matrix.std(axis=1)

    row = [df_hz, df_amp, vf]
    for i in range(13):
        row.append(float(mfcc_means[i]))
        row.append(float(mfcc_stds[i]))
    return np.array(row, dtype=np.float32)


# ============================================================
# SEGMENTATION — detect impacts and extract windows
# ============================================================

def segment_impacts(audio, sr):
    """
    Detect individual hammer impacts in a continuous recording.
    Returns list of 4410-sample numpy arrays (one per impact).
    """
    hop = 256
    frame_len = 512

    # Frame energy
    energy = np.array([
        np.sum(audio[i:i + frame_len] ** 2)
        for i in range(0, len(audio) - frame_len, hop)
    ])
    energy_norm = energy / (energy.max() + 1e-10)

    # Find peaks
    peaks, _ = find_peaks(energy_norm, height=0.1, distance=MIN_PEAK_DIST)

    segments = []
    for p in peaks:
        peak_sample = p * hop
        start = max(0, peak_sample - ONSET_OFFSET)
        end   = start + NUM_SAMPLES

        # If window overruns end of audio, shift it back
        if end > len(audio):
            end   = len(audio)
            start = max(0, end - NUM_SAMPLES)

        segment = audio[start:end]

        # Pad if still short (edge case)
        if len(segment) < NUM_SAMPLES:
            segment = np.pad(segment, (0, NUM_SAMPLES - len(segment)))

        segments.append(segment.astype(np.float32))

    return segments


# ============================================================
# LOAD BRIDGE DATA
# ============================================================

def load_bridge_data():
    """
    Load and segment all bridge WAV files.

    Returns:
      file_records  — list of dicts, one per original file:
                      {filename, true_label (0=Abnormal/1=Normal),
                       segments (list of waveforms),
                       features (2D array, one row per impact)}
    """
    labels_df = pd.read_csv(BRIDGE_LABELS)

    # Normalise filename: strip wav\ prefix, lower extension
    def clean_fn(s):
        return os.path.basename(s.replace("\\", "/"))

    labels_df["clean"] = labels_df["filename"].apply(clean_fn)

    # Map Normal->1 (Intact), Abnormal->0 (Cracked) to match lab encoding
    label_map = {"Normal": 1, "Abnormal": 0}

    file_records = []
    skipped = 0

    for _, row in labels_df.iterrows():
        fname      = row["clean"]
        true_label = label_map[row["State"]]
        fpath      = os.path.join(BRIDGE_DIR, fname)

        if not os.path.exists(fpath):
            # Try case-insensitive match
            all_files = {f.lower(): f for f in os.listdir(BRIDGE_DIR)}
            if fname.lower() in all_files:
                fpath = os.path.join(BRIDGE_DIR, all_files[fname.lower()])
            else:
                skipped += 1
                continue

        try:
            audio, sr = librosa.load(fpath, sr=TARGET_SR, mono=True)
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")
            skipped += 1
            continue

        segments = segment_impacts(audio, TARGET_SR)
        if len(segments) == 0:
            print(f"  Warning: no impacts detected in {fname}")
            skipped += 1
            continue

        features = np.stack([extract_features(s, TARGET_SR) for s in segments])

        file_records.append({
            "filename":   fname,
            "true_label": true_label,
            "n_impacts":  len(segments),
            "segments":   segments,
            "features":   features,
        })

    print(f"Loaded {len(file_records)} bridge files  ({skipped} skipped)")
    impact_counts = [r["n_impacts"] for r in file_records]
    print(f"  Impacts per file: min={min(impact_counts)}, "
          f"max={max(impact_counts)}, mean={np.mean(impact_counts):.1f}, "
          f"total={sum(impact_counts)}")
    return file_records


# ============================================================
# MAJORITY VOTE EVALUATION
# ============================================================

def majority_vote(per_impact_preds):
    """Return the majority class from a list of predictions."""
    counts = np.bincount(np.array(per_impact_preds, dtype=int))
    return int(np.argmax(counts))


def evaluate_model(name, file_records, predict_fn):
    """
    Run predict_fn on each impact, majority-vote per file,
    compute and return per-impact and per-file metrics.
    """
    per_impact_preds, per_impact_true = [], []
    per_file_preds,   per_file_true   = [], []

    for rec in file_records:
        impact_preds = predict_fn(rec["features"], rec["segments"])
        per_impact_preds.extend(impact_preds)
        per_impact_true.extend([rec["true_label"]] * len(impact_preds))

        file_pred = majority_vote(impact_preds)
        per_file_preds.append(file_pred)
        per_file_true.append(rec["true_label"])

    # Per-impact metrics
    ia_acc = accuracy_score(per_impact_true, per_impact_preds)
    ia_f1  = f1_score(per_impact_true, per_impact_preds, average="macro", zero_division=0)

    # Per-file (majority vote) metrics
    pf_acc = accuracy_score(per_file_true, per_file_preds)
    pf_f1  = f1_score(per_file_true, per_file_preds, average="macro", zero_division=0)

    class_names = ["Abnormal", "Normal"]
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"  Per-impact  — Accuracy: {ia_acc:.4f}  Macro F1: {ia_f1:.4f}  "
          f"(n={len(per_impact_preds)})")
    print(f"  Per-file MV — Accuracy: {pf_acc:.4f}  Macro F1: {pf_f1:.4f}  "
          f"(n={len(per_file_preds)})")
    print()
    print(classification_report(per_file_true, per_file_preds,
                                target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(per_file_true, per_file_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} — Bridge Zero-Shot\n"
                 f"Per-file Acc={pf_acc:.4f}  F1={pf_f1:.4f}")
    plt.tight_layout()
    safe_name = name.replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{safe_name}.png"), dpi=150)
    plt.close()

    return {
        "model":              name,
        "per_impact_acc":     round(ia_acc, 4),
        "per_impact_f1":      round(ia_f1,  4),
        "per_file_acc":       round(pf_acc, 4),
        "per_file_f1":        round(pf_f1,  4),
        "n_files":            len(per_file_preds),
        "n_impacts":          len(per_impact_preds),
    }


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim=29, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=64, stride=4, padding=32), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,  64,  kernel_size=32, stride=2, padding=16), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=16, stride=2, padding=8),  nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8,  stride=2, padding=4),  nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.classifier(self.gap(self.conv_blocks(x)))


# ============================================================
# MAIN
# ============================================================

def main():

    # ----------------------------------------------------------
    # 1. Load and segment bridge data
    # ----------------------------------------------------------
    print("=" * 60)
    print("LOADING BRIDGE DATA")
    print("=" * 60)
    file_records = load_bridge_data()

    n_normal   = sum(1 for r in file_records if r["true_label"] == 1)
    n_abnormal = sum(1 for r in file_records if r["true_label"] == 0)
    print(f"  Normal (Intact):   {n_normal} files")
    print(f"  Abnormal (Cracked): {n_abnormal} files")

    # ----------------------------------------------------------
    # 2. Load saved tabular models (no retraining)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING SAVED MODELS")
    print("=" * 60)

    rf      = joblib.load(RF_MODEL)
    print(f"  Random Forest loaded from: {RF_MODEL}")

    xgb_clf = joblib.load(XGB_MODEL)
    print(f"  XGBoost      loaded from:  {XGB_MODEL}")

    svm     = joblib.load(SVM_MODEL)
    print(f"  SVM          loaded from:  {SVM_MODEL}")

    # MLP scaler (fitted on train split of lab data)
    scaler  = joblib.load(MLP_SCALER)
    print(f"  MLP scaler   loaded from:  {MLP_SCALER}")

    device = torch.device("cpu")

    mlp_model = MLP(input_dim=len(FEATURE_COLS)).to(device)
    mlp_model.load_state_dict(torch.load(MLP_WEIGHTS, map_location=device))
    mlp_model.eval()
    print(f"  MLP loaded from:    {MLP_WEIGHTS}")

    cnn_model = CNN1D().to(device)
    cnn_model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device))
    cnn_model.eval()
    print(f"  1D CNN loaded from: {CNN_WEIGHTS}")

    # ----------------------------------------------------------
    # 4. Define predict functions for each model
    # ----------------------------------------------------------

    def predict_rf(features, segments):
        return rf.predict(features).tolist()

    def predict_xgb(features, segments):
        return xgb_clf.predict(features).tolist()

    def predict_svm(features, segments):
        return svm.predict(features).tolist()

    def predict_mlp(features, segments):
        X_scaled = scaler.transform(features).astype(np.float32)
        t = torch.tensor(X_scaled)
        with torch.no_grad():
            logits = mlp_model(t)
        return logits.argmax(1).numpy().tolist()

    def predict_cnn(features, segments):
        waves = np.stack(segments)                # (N, 4410)
        t = torch.tensor(waves).unsqueeze(1)      # (N, 1, 4410)
        with torch.no_grad():
            logits = cnn_model(t)
        return logits.argmax(1).numpy().tolist()

    # ----------------------------------------------------------
    # 5. Evaluate all models
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("ZERO-SHOT INFERENCE ON BRIDGE DATA")
    print("=" * 60)

    all_metrics = []
    models = [
        ("Random Forest",  predict_rf),
        ("XGBoost",        predict_xgb),
        ("SVM",            predict_svm),
        ("MLP",            predict_mlp),
        ("1D CNN",         predict_cnn),
    ]

    for name, predict_fn in models:
        metrics = evaluate_model(name, file_records, predict_fn)
        all_metrics.append(metrics)

    # ----------------------------------------------------------
    # 6. Summary table
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<16} {'Per-Impact Acc':>15} {'Per-Impact F1':>14} "
          f"{'Per-File Acc':>13} {'Per-File F1':>12}")
    print("-" * 72)
    for m in all_metrics:
        print(f"{m['model']:<16} {m['per_impact_acc']:>15.4f} {m['per_impact_f1']:>14.4f} "
              f"{m['per_file_acc']:>13.4f} {m['per_file_f1']:>12.4f}")

    # ----------------------------------------------------------
    # 7. Comparison bar chart
    # ----------------------------------------------------------
    model_names  = [m["model"]        for m in all_metrics]
    pf_accs      = [m["per_file_acc"] for m in all_metrics]
    pi_accs      = [m["per_impact_acc"] for m in all_metrics]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, pi_accs, width, label="Per-Impact Acc",  color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + width/2, pf_accs, width, label="Per-File (MV) Acc", color="darkorange", alpha=0.85)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline (50%)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Zero-Shot Transfer: Lab Models on Bridge Data\n"
                 "(Trained on lab specimens, tested on real bridge recordings)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "zero_shot_comparison.png"), dpi=150)
    plt.close()
    print(f"\nSaved: zero_shot_comparison.png")

    # ----------------------------------------------------------
    # 8. Save metrics JSON
    # ----------------------------------------------------------
    with open(os.path.join(OUTPUT_DIR, "zero_shot_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved: zero_shot_metrics.json")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
