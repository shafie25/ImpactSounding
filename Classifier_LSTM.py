# ============================================================
# Classifier — LSTM on MFCC Frame Sequences
# Acoustic Impact Hammer Testing: Cracked vs Intact
#
# Instead of collapsing MFCCs to mean/std statistics, each WAV
# is treated as a time sequence of MFCC frames:
#   shape (T=9, 13)  — 9 frames × 13 coefficients
#
# hop_length=512 (librosa default) on 4410 samples gives exactly
# T = 1 + 4410 // 512 = 9 frames per file (no padding needed).
#
# The LSTM captures temporal dynamics — how spectral content
# evolves from hammer impact through resonance to decay —
# which the tabular mean/std features discard.
#
# Architecture:
#   LSTM(13 → 64, 2 layers, dropout=0.3)
#   → last hidden state (64,)
#   → Linear(64→32) → ReLU → Dropout → Linear(32→2)
#
# Output: Classifier Results/LSTM/
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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================
# PARAMETERS
# ============================================================

DATA_DIR      = os.path.join("Specimens", "Class_sounds")
OUTPUT_DIR    = os.path.join("Classifier Results", "CrackDetection", "LSTM")
RANDOM_STATE  = 42

SAMPLE_RATE   = 22050
NUM_SAMPLES   = 4410        # 0.20s at 22050 Hz
N_MFCC        = 13
HOP_LENGTH    = 512         # default librosa hop → T = 9 frames
N_FFT         = 2048        # default librosa n_fft
SEQ_LEN       = 1 + NUM_SAMPLES // HOP_LENGTH   # = 9

HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3

# Class weights for imbalance (Cracked=1800, Intact=8236, N=10036)
N_CRACKED     = 1800
N_INTACT      = 8236
N_TOTAL       = N_CRACKED + N_INTACT
WEIGHT_CRACKED = N_TOTAL / (2 * N_CRACKED)   # ~2.788
WEIGHT_INTACT  = N_TOTAL / (2 * N_INTACT)    # ~0.609

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING — extract MFCC sequences
# ============================================================

def load_dataset(data_dir):
    """
    Walk Class_sounds/Cracked/ and Class_sounds/Intact/,
    load each WAV, pad/truncate to NUM_SAMPLES, then compute
    MFCC sequence with hop_length=HOP_LENGTH.

    Returns:
      X : (N, SEQ_LEN, N_MFCC) float32 — MFCC frame sequences
      y : (N,) int64             — 0=Cracked, 1=Intact
    """
    X, y = [], []
    for label_name, label_idx in [("Cracked", 0), ("Intact", 1)]:
        folder = os.path.join(data_dir, label_name)
        for root, _, files in os.walk(folder):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    audio, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
                    if len(audio) < NUM_SAMPLES:
                        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
                    else:
                        audio = audio[:NUM_SAMPLES]

                    # mfcc returns (N_MFCC, T) — transpose to (T, N_MFCC)
                    mfcc = librosa.feature.mfcc(
                        y=audio, sr=SAMPLE_RATE,
                        n_mfcc=N_MFCC,
                        hop_length=HOP_LENGTH,
                        n_fft=N_FFT,
                    )                          # (13, T)
                    mfcc = mfcc.T              # (T, 13)

                    # All files produce exactly SEQ_LEN frames; safety clip
                    mfcc = mfcc[:SEQ_LEN]
                    if mfcc.shape[0] < SEQ_LEN:
                        pad = np.zeros((SEQ_LEN - mfcc.shape[0], N_MFCC), dtype=np.float32)
                        mfcc = np.vstack([mfcc, pad])

                    X.append(mfcc.astype(np.float32))
                    y.append(label_idx)
                except Exception as e:
                    print(f"  Warning: could not load {fpath}: {e}")

    X = np.array(X, dtype=np.float32)   # (N, SEQ_LEN, N_MFCC)
    y = np.array(y, dtype=np.int64)
    print(f"Loaded {len(y)} files — Cracked: {(y==0).sum()}, Intact: {(y==1).sum()}")
    print(f"MFCC sequence shape per sample: ({SEQ_LEN}, {N_MFCC})")
    return X, y


# ============================================================
# DATASET WITH AUGMENTATION
# ============================================================

class MFCCDataset(Dataset):
    """
    Wraps (N, T, 13) MFCC arrays.
    When augment=True (training only):
      - Gaussian noise on MFCC values (std = 0.5% of feature std)
      - Time masking: randomly zero 1 consecutive frame
    """
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, T, 13)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()   # (T, 13)

        if self.augment:
            # Small MFCC noise
            noise = torch.randn_like(x) * (x.std() * 0.005 + 1e-6)
            x = x + noise

            # Randomly mask 1 time step
            mask_t = torch.randint(0, SEQ_LEN, (1,)).item()
            x[mask_t] = 0.0

        return x, self.y[idx]


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMClassifier(nn.Module):
    """
    Two-layer LSTM followed by a small MLP head.

    Input:  (B, T=9, 13)   — batch of MFCC sequences
    Output: (B, 2)          — logits for [Cracked, Intact]

    The last hidden state of the top LSTM layer is used as the
    sequence summary (not mean-pooling) because the end of the
    signal (decay) is most discriminative between crack states.
    """
    def __init__(self, input_size=N_MFCC, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        # x: (B, T, 13)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, hidden)
        last_hidden = h_n[-1]         # top layer: (B, hidden)
        return self.head(last_hidden)


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
    # 1. Load and extract MFCC sequences
    # ----------------------------------------------------------
    print("\n--- Loading WAV files and extracting MFCC sequences ---")
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
    # 0.125 of 0.80 = 0.10 of total → 80/10/10

    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"  Train — Cracked: {(y_train==0).sum()}, Intact: {(y_train==1).sum()}")
    print(f"  Val   — Cracked: {(y_val==0).sum()},   Intact: {(y_val==1).sum()}")
    print(f"  Test  — Cracked: {(y_test==0).sum()},  Intact: {(y_test==1).sum()}")

    # ----------------------------------------------------------
    # 3. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_ds = MFCCDataset(X_train, y_train, augment=True)
    val_ds   = MFCCDataset(X_val,   y_val,   augment=False)
    test_ds  = MFCCDataset(X_test,  y_test,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------
    # 4. Model, loss, optimizer, scheduler
    # ----------------------------------------------------------
    model = LSTMClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"LSTM: {NUM_LAYERS} layers, hidden={HIDDEN_SIZE}, input={N_MFCC}, seq_len={SEQ_LEN}")

    class_weights = torch.tensor(
        [WEIGHT_CRACKED, WEIGHT_INTACT], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    # ----------------------------------------------------------
    # 5. Training loop — checkpoint by best val F1 (Cracked)
    # ----------------------------------------------------------
    print("\n--- Training ---")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_f1 = 0.0
    best_epoch  = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_preds, vl_labels = evaluate(model, val_loader, criterion, device)

        vl_f1 = f1_score(vl_labels, vl_preds, pos_label=0, zero_division=0)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f} | "
                  f"lr={lr_now:.2e}")

    print(f"\nBest checkpoint: epoch {best_epoch} (val F1={best_val_f1:.4f})")

    # ----------------------------------------------------------
    # 6. Test evaluation (best checkpoint)
    # ----------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"),
                                     map_location=device))
    _, _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    test_acc  = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, pos_label=0, zero_division=0)
    test_rec  = recall_score(test_labels, test_preds, pos_label=0, zero_division=0)
    test_f1   = f1_score(test_labels, test_preds, pos_label=0, zero_division=0)

    print("\n--- Test Results ---")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}  (Cracked class)")
    print(f"  Recall:    {test_rec:.4f}   (Cracked class)")
    print(f"  F1:        {test_f1:.4f}    (Cracked class)")
    print()
    print(classification_report(test_labels, test_preds,
                                target_names=["Cracked", "Intact"]))

    # Val metrics at best checkpoint (used as cv_* in Classifier_Comparison)
    _, _, val_preds_best, val_labels_best = evaluate(model, val_loader, criterion, device)
    cv_acc  = accuracy_score(val_labels_best, val_preds_best)
    cv_prec = precision_score(val_labels_best, val_preds_best, pos_label=0, zero_division=0)
    cv_rec  = recall_score(val_labels_best, val_preds_best, pos_label=0, zero_division=0)
    cv_f1   = f1_score(val_labels_best, val_preds_best, pos_label=0, zero_division=0)

    # ----------------------------------------------------------
    # 7. Save metrics.json
    # ----------------------------------------------------------
    metrics = {
        "model":          "LSTM",
        "test_accuracy":  round(test_acc,  4),
        "test_precision": round(test_prec, 4),
        "test_recall":    round(test_rec,  4),
        "test_f1":        round(test_f1,   4),
        "cv_accuracy":    round(cv_acc,    4),
        "cv_precision":   round(cv_prec,   4),
        "cv_recall":      round(cv_rec,    4),
        "cv_f1":          round(cv_f1,     4),
        "best_epoch":     best_epoch,
        "best_val_f1":    round(best_val_f1, 4),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

    # ----------------------------------------------------------
    # 8. Confusion matrix
    # ----------------------------------------------------------
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cracked", "Intact"],
                yticklabels=["Cracked", "Intact"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"LSTM — Confusion Matrix\n"
                 f"Acc={test_acc:.4f}  Prec={test_prec:.4f}  "
                 f"Rec={test_rec:.4f}  F1={test_f1:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")

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

    plt.suptitle("LSTM on MFCC Sequences — Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved training_curves.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
