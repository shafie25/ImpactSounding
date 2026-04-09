# ============================================================
# Classifier_Width_MLP.py
# Crack Width Classification — MLP on Tabular Features
# Level 2a: 0.2mm / 0.4mm / 0.6mm (3-class)
#
# Trains on Cracked samples only (1,800 files).
# Labels derived from Specimen column in Labeled_Features.xlsx:
#   specimen_X_1 → 0.2mm (label 0)
#   specimen_X_2 → 0.4mm (label 1)
#   specimen_X_3 → 0.6mm (label 2)
#
# Classes are perfectly balanced (600 per class) — no class
# weighting needed.
#
# Same architecture as CrackDetection MLP but with 3 output
# classes and no class weighting.
#
# Output: Classifier Results/CrackWidth/MLP/
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)


# ============================================================
# PARAMETERS
# ============================================================

FEATURES_FILE = "Labeled_Features.xlsx"
OUTPUT_DIR    = os.path.join("Classifier Results", "CrackWidth", "MLP")
RANDOM_STATE  = 42

CLASS_NAMES   = ["0.2mm", "0.4mm", "0.6mm"]
NUM_CLASSES   = 3

BATCH_SIZE    = 32
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-3
DROPOUT       = 0.3

FEATURE_COLS  = (
    ["Df_Hz", "Df_Amplitude", "Vf"] +
    [col for i in range(1, 14) for col in (f"MFCC_{i}_mean", f"MFCC_{i}_std")]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_width_data(path):
    """
    Returns X, y, is_test where is_test marks series-2 rows as test.
    Series 1 (specimen_1_X) → train/val
    Series 2 (specimen_2_X) → test   [specimen-level split]
    """
    df = pd.read_excel(path)
    df = df[df["Label"] == "Cracked"].copy()

    width_digit = df["Specimen"].str.extract(r"specimen_\d+_(\d+)")[0].astype(int)
    series      = df["Specimen"].str.extract(r"specimen_(\d+)_\d+")[0].astype(int)

    y       = (width_digit - 1).values.astype(np.int64)
    X       = df[FEATURE_COLS].values.astype(np.float32)
    is_test = (series == 2).values

    print(f"Loaded {len(y)} Cracked samples")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == i).sum()} samples")
    return X, y, is_test


# ============================================================
# DATASET
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n--- Loading crack width data ---")
    X, y, is_test = load_width_data(FEATURES_FILE)

    # ----------------------------------------------------------
    # 2. Specimen-level split  (Series 1 → train/val, Series 2 → test)
    # ----------------------------------------------------------
    X_trainval, y_trainval = X[~is_test], y[~is_test]
    X_test,     y_test     = X[is_test],  y[is_test]

    # Random 90/10 val split within series 1 (test boundary is already clean)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.10, stratify=y_trainval, random_state=RANDOM_STATE
    )
    print(f"\nSpecimen-level split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"  (series 1 -> train/val, series 2 -> test)")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Train {name}: {(y_train==i).sum()}  "
              f"Val {name}: {(y_val==i).sum()}  "
              f"Test {name}: {(y_test==i).sum()}")

    # ----------------------------------------------------------
    # 3. StandardScaler — fit on train only
    # ----------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ----------------------------------------------------------
    # 4. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_loader = DataLoader(TabularDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TabularDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TabularDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------
    # 5. Model, loss, optimizer, scheduler
    # ----------------------------------------------------------
    model = MLP(input_dim=X_train.shape[1]).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()   # balanced classes — no weighting needed
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    # ----------------------------------------------------------
    # 6. Training loop — checkpoint by best val macro F1
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
    # 7. Test evaluation
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
    # 8. Save metrics.json
    # ----------------------------------------------------------
    metrics = {
        "model":         "MLP",
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
    # 9. Confusion matrix
    # ----------------------------------------------------------
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"MLP — Crack Width Confusion Matrix\n"
                 f"Acc={acc:.4f}  Macro F1={f1_macro:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # ----------------------------------------------------------
    # 10. Training curves
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

    plt.suptitle("MLP on Tabular Features — Crack Width Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved: training_curves.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
