# ============================================================
# Classifier_Width_1DCNN.py
# Crack Width Classification — 1D CNN on Raw Waveforms
# Level 2a: 0.2mm / 0.4mm / 0.6mm (3-class)
#
# Trains on Cracked WAV files only (1,800 files).
# Labels derived from folder name:
#   specimen_X_1... → 0.2mm (label 0)
#   specimen_X_2... → 0.4mm (label 1)
#   specimen_X_3... → 0.6mm (label 2)
#
# Classes are perfectly balanced (600 per class) — no class
# weighting needed.
#
# Same architecture as CrackDetection CNN_1D. Stronger
# augmentation to compensate for the smaller dataset (1,800
# vs 10,036 files).
#
# Output: Classifier Results/CrackWidth/CNN_1D/
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)


# ============================================================
# PARAMETERS
# ============================================================

DATA_DIR     = os.path.join("Specimens", "Class_sounds", "Cracked")
OUTPUT_DIR   = os.path.join("Classifier Results", "CrackWidth", "CNN_1D")
RANDOM_STATE = 42

SAMPLE_RATE  = 22050
NUM_SAMPLES  = 4410       # 0.20s

CLASS_NAMES  = ["0.2mm", "0.4mm", "0.6mm"]
NUM_CLASSES  = 3

BATCH_SIZE   = 32          # smaller batch — less data
NUM_EPOCHS   = 80
LEARNING_RATE = 1e-3
DROPOUT      = 0.4         # slightly stronger than Level 1 (0.3) — less data

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(data_dir):
    """
    Walk Specimens/Class_sounds/Cracked/, load each WAV,
    extract width label from folder name.

    Folder pattern: specimen_X_Y<SoundData>...
      Y=1 → 0.2mm → label 0
      Y=2 → 0.4mm → label 1
      Y=3 → 0.6mm → label 2
    """
    import re
    X, y = [], []
    pattern = re.compile(r"specimen_\d+_(\d+)")

    for folder in sorted(os.listdir(data_dir)):
        match = pattern.search(folder)
        if not match:
            continue
        width_digit = int(match.group(1))
        label = width_digit - 1   # 1→0, 2→1, 3→2

        folder_path = os.path.join(data_dir, folder, "Sounds")
        if not os.path.isdir(folder_path):
            folder_path = os.path.join(data_dir, folder)

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(folder_path, fname)
            try:
                audio, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
                if len(audio) < NUM_SAMPLES:
                    audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
                else:
                    audio = audio[:NUM_SAMPLES]
                X.append(audio.astype(np.float32))
                y.append(label)
            except Exception as e:
                print(f"  Warning: could not load {fpath}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Loaded {len(y)} files")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == i).sum()} files")
    return X, y


# ============================================================
# DATASET WITH AUGMENTATION
# ============================================================

class WaveformDataset(Dataset):
    """
    Augmentation (training only):
      - Gaussian noise: std = 1% of signal std  (stronger than Level 1)
      - Random circular time shift: ±10%         (stronger than Level 1)
      - Random amplitude scale: ×U(0.85, 1.15)  (new — helps with less data)
    """
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()

        if self.augment:
            # Gaussian noise
            noise = torch.randn_like(x) * (x.std() * 0.01)
            x = x + noise

            # Random time shift ±10%
            max_shift = int(0.10 * NUM_SAMPLES)
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            x = torch.roll(x, shift)

            # Random amplitude scale
            scale = 0.85 + torch.rand(1).item() * 0.30   # U(0.85, 1.15)
            x = x * scale

        return x.unsqueeze(0), self.y[idx]   # (1, 4410)


# ============================================================
# MODEL  (same architecture as CrackDetection CNN_1D)
# ============================================================

class CNN1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
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
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.conv_blocks(x)))


# ============================================================
# TRAINING / EVALUATION
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)
        total_loss += loss.item() * len(y)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n--- Loading WAV files ---")
    X, y = load_dataset(DATA_DIR)

    # ----------------------------------------------------------
    # 2. Stratified 80/10/10 split
    # ----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"\nSplit: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Train {name}: {(y_train==i).sum()}  "
              f"Val {name}: {(y_val==i).sum()}  "
              f"Test {name}: {(y_test==i).sum()}")

    # ----------------------------------------------------------
    # 3. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_loader = DataLoader(WaveformDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(WaveformDataset(X_val,   y_val,   augment=False),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(WaveformDataset(X_test,  y_test,  augment=False),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------
    # 4. Model, loss, optimizer, scheduler
    # ----------------------------------------------------------
    model = CNN1D().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()   # balanced classes — no weighting needed
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5
    )

    # ----------------------------------------------------------
    # 5. Training loop — checkpoint by best val macro F1
    # ----------------------------------------------------------
    print("\n--- Training ---")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_f1 = 0.0
    best_epoch  = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_preds, vl_labels = evaluate(model, val_loader, criterion, device)

        vl_f1 = f1_score(vl_labels, vl_preds, average="macro", zero_division=0)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f} | "
                  f"lr={lr_now:.2e}")

    print(f"\nBest checkpoint: epoch {best_epoch} (val macro F1={best_val_f1:.4f})")

    # ----------------------------------------------------------
    # 6. Test evaluation
    # ----------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"),
                                     map_location=device))
    _, _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    acc      = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    f1_per   = f1_score(test_labels, test_preds, average=None, zero_division=0)

    print("\n--- Test Results ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1_macro:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  F1 ({name}): {f1_per[i]:.4f}")
    print()
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    # Val metrics at best checkpoint
    _, _, val_preds_best, val_labels_best = evaluate(model, val_loader, criterion, device)
    cv_acc = accuracy_score(val_labels_best, val_preds_best)
    cv_f1  = f1_score(val_labels_best, val_preds_best, average="macro", zero_division=0)

    # ----------------------------------------------------------
    # 7. Save metrics.json
    # ----------------------------------------------------------
    metrics = {
        "model":         "CNN_1D",
        "task":          "CrackWidth",
        "test_accuracy": round(acc,      4),
        "test_f1_macro": round(f1_macro, 4),
        "test_f1_02mm":  round(float(f1_per[0]), 4),
        "test_f1_04mm":  round(float(f1_per[1]), 4),
        "test_f1_06mm":  round(float(f1_per[2]), 4),
        "cv_accuracy":   round(cv_acc,   4),
        "cv_f1_macro":   round(cv_f1,    4),
        "best_epoch":    best_epoch,
        "best_val_f1":   round(best_val_f1, 4),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: metrics.json")

    # ----------------------------------------------------------
    # 8. Confusion matrix
    # ----------------------------------------------------------
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"1D CNN — Crack Width Confusion Matrix\n"
                 f"Acc={acc:.4f}  Macro F1={f1_macro:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # ----------------------------------------------------------
    # 9. Training curves
    # ----------------------------------------------------------
    epochs = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].axvline(best_epoch, color="red", linestyle="--", linewidth=0.8,
                    label=f"Best epoch {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    axes[1].axvline(best_epoch, color="red", linestyle="--", linewidth=0.8,
                    label=f"Best epoch {best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()

    plt.suptitle("1D CNN on Raw Waveforms — Crack Width Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved: training_curves.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
