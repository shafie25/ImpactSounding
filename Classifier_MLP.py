# ============================================================
# Classifier — MLP on Tabular Features
# Acoustic Impact Hammer Testing: Cracked vs Intact
#
# Uses the same 29 pre-extracted features as SVM/XGBoost/RF
# (Df, Vf, 13 MFCC mean, 13 MFCC std) from Labeled_Features.xlsx.
# No audio loading — purely tabular input.
#
# Features are StandardScaler-normalized before training
# (unlike tree-based models, MLPs are scale-sensitive).
#
# Architecture:
#   Linear(29→128) → BN → ReLU → Dropout(0.3)
#   Linear(128→64) → BN → ReLU → Dropout(0.3)
#   Linear(64→2)
#
# This is the fairest NN vs classical comparison: identical
# features, different learner. Tells us whether the feature
# space is linearly separable (SVM wins) or needs nonlinear
# depth to extract value (MLP wins).
#
# Output: Classifier Results/MLP/
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================
# PARAMETERS
# ============================================================

FEATURES_FILE = "Labeled_Features.xlsx"
OUTPUT_DIR    = os.path.join("Classifier Results", "CrackDetection", "MLP")
RANDOM_STATE  = 42

BATCH_SIZE    = 64
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-3
DROPOUT       = 0.3

# Class weights for imbalance (Cracked=1800, Intact=8236, N=10036)
N_CRACKED      = 1800
N_INTACT       = 8236
N_TOTAL        = N_CRACKED + N_INTACT
WEIGHT_CRACKED = N_TOTAL / (2 * N_CRACKED)   # ~2.788
WEIGHT_INTACT  = N_TOTAL / (2 * N_INTACT)    # ~0.609

# Feature columns — same 29 used by SVM/XGBoost/RF
FEATURE_COLS = (
    ["Df_Hz", "Df_Amplitude", "Vf"] +
    [col for i in range(1, 14) for col in (f"MFCC_{i}_mean", f"MFCC_{i}_std")]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_features(path):
    df = pd.read_excel(path)
    # Label column: 'Label' with values 'Cracked'/'Intact'
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = (df["Label"] == "Intact").astype(np.int64).values   # 0=Cracked, 1=Intact
    print(f"Loaded {len(y)} samples — Cracked: {(y==0).sum()}, Intact: {(y==1).sum()}")
    print(f"Features: {X.shape[1]} columns")
    return X, y


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
    """
    Two hidden-layer MLP with BatchNorm and Dropout.

    BatchNorm stabilises training on heterogeneous tabular
    features (Df is in Hz, MFCCs are dimensionless dB-like
    values — very different scales even after StandardScaler).

    Input:  (B, 29)
    Output: (B, 2) logits
    """
    def __init__(self, input_dim=29, dropout=DROPOUT):
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
            nn.Linear(64, 2),
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
    print(f"PyTorch: {torch.__version__}")

    # ----------------------------------------------------------
    # 1. Load features
    # ----------------------------------------------------------
    print(f"\n--- Loading features from {FEATURES_FILE} ---")
    X, y = load_features(FEATURES_FILE)

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
    # 3. StandardScaler — fit on train only
    # ----------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ----------------------------------------------------------
    # 4. Datasets & DataLoaders
    # ----------------------------------------------------------
    train_ds = TabularDataset(X_train, y_train)
    val_ds   = TabularDataset(X_val,   y_val)
    test_ds  = TabularDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------
    # 5. Model, loss, optimizer, scheduler
    # ----------------------------------------------------------
    input_dim = X_train.shape[1]
    print(f"Input dim: {input_dim} features")
    model = MLP(input_dim=input_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    class_weights = torch.tensor(
        [WEIGHT_CRACKED, WEIGHT_INTACT], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    # ----------------------------------------------------------
    # 6. Training loop — checkpoint by best val F1 (Cracked)
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

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"Val loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f} | "
                  f"lr={lr_now:.2e}")

    print(f"\nBest checkpoint: epoch {best_epoch} (val F1={best_val_f1:.4f})")

    # ----------------------------------------------------------
    # 7. Test evaluation (best checkpoint)
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

    # Val metrics at best checkpoint
    _, _, val_preds_best, val_labels_best = evaluate(model, val_loader, criterion, device)
    cv_acc  = accuracy_score(val_labels_best, val_preds_best)
    cv_prec = precision_score(val_labels_best, val_preds_best, pos_label=0, zero_division=0)
    cv_rec  = recall_score(val_labels_best, val_preds_best, pos_label=0, zero_division=0)
    cv_f1   = f1_score(val_labels_best, val_preds_best, pos_label=0, zero_division=0)

    # ----------------------------------------------------------
    # 8. Save metrics.json
    # ----------------------------------------------------------
    metrics = {
        "model":          "MLP",
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
    # 9. Confusion matrix
    # ----------------------------------------------------------
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cracked", "Intact"],
                yticklabels=["Cracked", "Intact"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"MLP — Confusion Matrix\n"
                 f"Acc={test_acc:.4f}  Prec={test_prec:.4f}  "
                 f"Rec={test_rec:.4f}  F1={test_f1:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")

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

    plt.suptitle("MLP on Tabular Features — Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved training_curves.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
