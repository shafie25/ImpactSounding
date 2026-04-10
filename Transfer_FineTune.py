# ============================================================
# Transfer_FineTune.py
# Frozen-Backbone Transfer Learning — Bridge Dataset (JapanDataset2)
#
# Uses the 1D CNN trained on lab specimens. All four conv blocks
# are frozen; only the classifier head (Linear(256->64)->Linear(64->2))
# is retrained on a small portion of bridge data.
#
# Why frozen backbone only:
#   - The conv layers learned acoustic waveform representations
#     (onset shape, decay, frequency content) from lab data.
#   - We have very few bridge samples (bridge is out-of-domain).
#   - Retraining the entire network on 160 files would overfit
#     and destroy the lab-learned representations.
#   - Only adapting the head lets the network map lab features
#     to the bridge domain's decision boundary.
#
# Data split (location-level to prevent L/R mic leakage):
#   - 50 unique locations, each producing 4 files:
#     Normal_L, Normal_R, Abnormal_L, Abnormal_R
#   - Split at location level: train=35, val=5, test=10
#   - All 4 files from the same location stay together
#
# Evaluation: majority vote across impacts per file
#
# Output: Classifier Results/TransferLearning/FineTune/
# ============================================================

import os
import re
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)


# ============================================================
# PARAMETERS
# ============================================================

BRIDGE_DIR    = "JapanDataset2"
BRIDGE_LABELS = "JapanDataset2Labels.csv"
CNN_WEIGHTS   = os.path.join("Classifier Results", "CrackDetection", "CNN_1D", "best_model.pt")
OUTPUT_DIR    = os.path.join("Classifier Results", "TransferLearning", "FineTune")

RANDOM_STATE  = 42
TARGET_SR     = 22050
NUM_SAMPLES   = 4410          # 0.20s at 22050 Hz
ONSET_OFFSET  = 220           # 10ms before peak
MIN_PEAK_DIST = int(0.25 * TARGET_SR / 256)

# Fine-tune hyperparameters
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LR_HEAD       = 1e-3          # learning rate for head only

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL  (must match Classifier_1DCNN.py exactly)
# ============================================================

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
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.conv_blocks(x)))


# ============================================================
# AUDIO UTILITIES  (same as Transfer_ZeroShot.py)
# ============================================================

def segment_impacts(audio, sr):
    """Detect individual impacts; return list of 4410-sample arrays."""
    hop = 256
    frame_len = 512
    energy = np.array([
        np.sum(audio[i:i + frame_len] ** 2)
        for i in range(0, len(audio) - frame_len, hop)
    ])
    energy_norm = energy / (energy.max() + 1e-10)
    peaks, _ = find_peaks(energy_norm, height=0.1, distance=MIN_PEAK_DIST)

    segments = []
    for p in peaks:
        peak_sample = p * hop
        start = max(0, peak_sample - ONSET_OFFSET)
        end   = start + NUM_SAMPLES
        if end > len(audio):
            end   = len(audio)
            start = max(0, end - NUM_SAMPLES)
        segment = audio[start:end]
        if len(segment) < NUM_SAMPLES:
            segment = np.pad(segment, (0, NUM_SAMPLES - len(segment)))
        segments.append(segment.astype(np.float32))
    return segments


def load_bridge_records():
    """
    Load all bridge WAV files, segment into individual impacts.
    Returns list of dicts with keys:
        filename, location_id, true_label, segments
    """
    labels_df = pd.read_csv(BRIDGE_LABELS)

    def clean_fn(s):
        return os.path.basename(s.replace("\\", "/"))

    labels_df["clean"] = labels_df["filename"].apply(clean_fn)

    # Extract location id: "3-1_L.wav" -> location "3-1"
    def get_location(fn):
        m = re.match(r"(.+?)_[LR]\.wav", fn, re.IGNORECASE)
        return m.group(1) if m else fn

    labels_df["location"] = labels_df["clean"].apply(get_location)
    label_map = {"Normal": 1, "Abnormal": 0}

    records = []
    skipped = 0

    for _, row in labels_df.iterrows():
        fname      = row["clean"]
        true_label = label_map[row["State"]]
        location   = row["location"]
        fpath      = os.path.join(BRIDGE_DIR, fname)

        if not os.path.exists(fpath):
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

        segs = segment_impacts(audio, TARGET_SR)
        if len(segs) == 0:
            print(f"  Warning: no impacts detected in {fname}")
            skipped += 1
            continue

        records.append({
            "filename":   fname,
            "location":   location,
            "true_label": true_label,
            "segments":   segs,
        })

    print(f"Loaded {len(records)} files  ({skipped} skipped)")
    return records


# ============================================================
# DATASET
# ============================================================

class ImpactDataset(Dataset):
    def __init__(self, records):
        self.waves  = []
        self.labels = []
        for rec in records:
            for seg in rec["segments"]:
                self.waves.append(seg)
                self.labels.append(rec["true_label"])
        self.waves  = np.array(self.waves,  dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.waves[idx]).unsqueeze(0)   # (1, 4410)
        y = torch.tensor(self.labels[idx])
        return x, y


# ============================================================
# MAJORITY VOTE EVALUATION
# ============================================================

def evaluate_majority_vote(model, records, device):
    model.eval()
    per_file_preds = []
    per_file_true  = []

    with torch.no_grad():
        for rec in records:
            waves = np.stack(rec["segments"]).astype(np.float32)
            t = torch.tensor(waves).unsqueeze(1).to(device)  # (N, 1, 4410)
            logits = model(t)
            preds  = logits.argmax(1).cpu().numpy().tolist()
            counts = np.bincount(preds, minlength=2)
            file_pred = int(np.argmax(counts))
            per_file_preds.append(file_pred)
            per_file_true.append(rec["true_label"])

    return np.array(per_file_true), np.array(per_file_preds)


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cpu")
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # ----------------------------------------------------------
    # 1. Load bridge data
    # ----------------------------------------------------------
    print("=" * 60)
    print("LOADING BRIDGE DATA")
    print("=" * 60)
    all_records = load_bridge_records()

    # ----------------------------------------------------------
    # 2. Location-level split (35 / 5 / 10 locations)
    # ----------------------------------------------------------
    locations = sorted(set(r["location"] for r in all_records))
    print(f"Unique locations: {len(locations)}")

    rng = np.random.default_rng(RANDOM_STATE)
    loc_arr = np.array(locations)
    rng.shuffle(loc_arr)

    n_test  = 10
    n_val   = 5
    n_train = len(loc_arr) - n_test - n_val   # 35

    train_locs = set(loc_arr[:n_train])
    val_locs   = set(loc_arr[n_train:n_train + n_val])
    test_locs  = set(loc_arr[n_train + n_val:])

    train_recs = [r for r in all_records if r["location"] in train_locs]
    val_recs   = [r for r in all_records if r["location"] in val_locs]
    test_recs  = [r for r in all_records if r["location"] in test_locs]

    print(f"Location split: train={len(train_locs)}  val={len(val_locs)}  test={len(test_locs)}")
    print(f"File split:     train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")

    train_normal   = sum(1 for r in train_recs if r["true_label"] == 1)
    train_abnormal = sum(1 for r in train_recs if r["true_label"] == 0)
    print(f"Train: Normal={train_normal}  Abnormal={train_abnormal}")

    # ----------------------------------------------------------
    # 3. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_ds = ImpactDataset(train_recs)
    val_ds   = ImpactDataset(val_recs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Impact counts: train={len(train_ds)}  val={len(val_ds)}  "
          f"test={sum(len(r['segments']) for r in test_recs)}")

    # ----------------------------------------------------------
    # 4. Load pretrained model, freeze conv blocks
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING PRETRAINED CNN AND FREEZING CONV BLOCKS")
    print("=" * 60)

    model = CNN1D().to(device)
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device))
    print(f"Weights loaded from: {CNN_WEIGHTS}")

    # Freeze all conv_blocks parameters
    for param in model.conv_blocks.parameters():
        param.requires_grad = False
    for param in model.gap.parameters():
        param.requires_grad = False

    frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters:   {frozen:,}")
    print(f"Trainable parameters:{trainable:,}  (classifier head only)")

    # ----------------------------------------------------------
    # 5. Loss and optimizer (head parameters only)
    # ----------------------------------------------------------
    # Bridge data is balanced (100 Normal / 100 Abnormal) so no
    # class weighting needed.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    # ----------------------------------------------------------
    # 6. Training loop — checkpoint by best val accuracy
    # ----------------------------------------------------------
    print("\n--- Fine-tuning classifier head ---")
    best_val_acc = 0.0
    best_epoch   = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += len(y)
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        # Val
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                vl_loss    += loss.item() * len(y)
                vl_correct += (logits.argmax(1) == y).sum().item()
                vl_total   += len(y)
        vl_loss /= vl_total
        vl_acc   = vl_correct / vl_total

        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
                  f"lr={lr_now:.2e}")

    print(f"\nBest checkpoint: epoch {best_epoch} (val acc={best_val_acc:.4f})")

    # ----------------------------------------------------------
    # 7. Test evaluation (majority vote, best checkpoint)
    # ----------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"),
                                     map_location=device))

    y_true, y_pred = evaluate_majority_vote(model, test_recs, device)
    class_names = ["Abnormal", "Normal"]

    test_acc = accuracy_score(y_true, y_pred)
    test_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n--- Test Results (per-file majority vote) ---")
    print(f"  Files: {len(y_true)}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {test_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Zero-shot baseline on same test set (frozen conv, original head)
    model_zs = CNN1D().to(device)
    model_zs.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device))
    model_zs.eval()
    yz_true, yz_pred = evaluate_majority_vote(model_zs, test_recs, device)
    zs_acc = accuracy_score(yz_true, yz_pred)
    zs_f1  = f1_score(yz_true, yz_pred, average="macro", zero_division=0)
    print(f"Zero-shot baseline on same test split: acc={zs_acc:.4f}  f1={zs_f1:.4f}")

    # ----------------------------------------------------------
    # 8. Save metrics
    # ----------------------------------------------------------
    results = {
        "model":               "1D CNN Frozen Backbone",
        "train_locations":     n_train,
        "val_locations":       n_val,
        "test_locations":      n_test,
        "trainable_params":    trainable,
        "frozen_params":       frozen,
        "best_epoch":          best_epoch,
        "best_val_acc":        round(best_val_acc, 4),
        "test_accuracy":       round(test_acc, 4),
        "test_macro_f1":       round(test_f1,  4),
        "zeroshot_acc":        round(zs_acc, 4),
        "zeroshot_f1":         round(zs_f1,  4),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: metrics.json")

    # ----------------------------------------------------------
    # 9. Confusion matrix
    # ----------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Frozen-Backbone CNN — Bridge Test\n"
                 f"Acc={test_acc:.4f}  F1={test_f1:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # ----------------------------------------------------------
    # 10. Training curves
    # ----------------------------------------------------------
    epochs = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].axvline(best_epoch, color="red", linestyle="--", linewidth=0.8,
                    label=f"Best epoch {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    axes[1].axvline(best_epoch, color="red", linestyle="--", linewidth=0.8,
                    label=f"Best epoch {best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.suptitle("Frozen-Backbone CNN Fine-Tune on Bridge Data")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved: training_curves.png")

    # ----------------------------------------------------------
    # 11. Comparison bar chart: zero-shot vs frozen-backbone
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    models_c   = ["Zero-Shot", "Frozen Backbone\n(head retrained)"]
    accs       = [zs_acc, test_acc]
    f1s        = [zs_f1,  test_f1]
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",  color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s,  width, label="Macro F1",  color="darkorange", alpha=0.85)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(models_c)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("1D CNN: Zero-Shot vs Frozen-Backbone Fine-Tune\n(same test locations)")
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison.png"), dpi=150)
    plt.close()
    print("Saved: comparison.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
