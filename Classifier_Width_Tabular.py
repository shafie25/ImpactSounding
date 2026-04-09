# ============================================================
# Classifier_Width_Tabular.py
# Crack Width Classification — RF, XGBoost, SVM
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
# Uses same 29 tabular features as CrackDetection classifiers.
# StandardScaler applied for SVM only.
#
# Output: Classifier Results/CrackWidth/{RandomForest,XGBoost,SVM}/
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

import xgboost as xgb
from sklearn.svm import SVC


# ============================================================
# PARAMETERS
# ============================================================

FEATURES_FILE = "Labeled_Features.xlsx"
RESULTS_ROOT  = os.path.join("Classifier Results", "CrackWidth")
RANDOM_STATE  = 42

CLASS_NAMES   = ["0.2mm", "0.4mm", "0.6mm"]

FEATURE_COLS  = (
    ["Df_Hz", "Df_Amplitude", "Vf"] +
    [col for i in range(1, 14) for col in (f"MFCC_{i}_mean", f"MFCC_{i}_std")]
)

for d in [
    os.path.join(RESULTS_ROOT, "RandomForest"),
    os.path.join(RESULTS_ROOT, "XGBoost"),
    os.path.join(RESULTS_ROOT, "SVM"),
]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_width_data(path):
    """
    Load Labeled_Features.xlsx, filter to Cracked rows only,
    derive width labels and specimen series from the Specimen column.

    specimen_X_1 → label 0 (0.2mm)
    specimen_X_2 → label 1 (0.4mm)
    specimen_X_3 → label 2 (0.6mm)

    Series 1 (specimen_1_X) → train
    Series 2 (specimen_2_X) → test   [specimen-level split]
    """
    df = pd.read_excel(path)
    df = df[df["Label"] == "Cracked"].copy()

    width_digit = df["Specimen"].str.extract(r"specimen_\d+_(\d+)")[0].astype(int)
    series      = df["Specimen"].str.extract(r"specimen_(\d+)_\d+")[0].astype(int)

    y       = (width_digit - 1).values   # 1→0, 2→1, 3→2
    X       = df[FEATURE_COLS].values.astype(np.float32)
    is_test = (series == 2).values       # True → test set

    print(f"Loaded {len(y)} Cracked samples")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == i).sum()} samples")
    return X, y, is_test


# ============================================================
# HELPERS
# ============================================================

def save_metrics(metrics, output_dir):
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: metrics.json")


def plot_confusion_matrix(labels, preds, model_name, acc, f1_macro, output_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Crack Width Confusion Matrix\n"
                 f"Acc={acc:.4f}  Macro F1={f1_macro:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  Saved: confusion_matrix.png")


def plot_cv_scores(cv_results, model_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    folds = range(1, 6)

    axes[0].bar(folds, cv_results["test_accuracy"], color="steelblue", alpha=0.8)
    axes[0].axhline(cv_results["test_accuracy"].mean(), color="red",
                    linestyle="--", label=f"Mean={cv_results['test_accuracy'].mean():.4f}")
    axes[0].set_title("5-Fold CV — Accuracy")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].bar(folds, cv_results["test_f1_macro"], color="darkorange", alpha=0.8)
    axes[1].axhline(cv_results["test_f1_macro"].mean(), color="red",
                    linestyle="--", label=f"Mean={cv_results['test_f1_macro'].mean():.4f}")
    axes[1].set_title("5-Fold CV — Macro F1")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    plt.suptitle(f"{model_name} — Crack Width 5-Fold CV")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_scores.png"), dpi=150)
    plt.close()
    print("  Saved: cv_scores.png")


def evaluate_and_save(model, X_train, y_train, X_test, y_test,
                      model_name, output_dir):
    """Fit, evaluate test set + 5-fold CV, save all outputs."""

    # -- Test set --
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc      = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_per   = f1_score(y_test, preds, average=None, zero_division=0)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Test Macro F1 : {f1_macro:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  F1 ({name})    : {f1_per[i]:.4f}")
    print()
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    # -- 5-fold CV --
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy",
                 "f1_macro": "f1_macro"},
        return_train_score=False,
    )

    cv_acc     = cv_results["test_accuracy"].mean()
    cv_f1      = cv_results["test_f1_macro"].mean()
    print(f"  CV Accuracy   : {cv_acc:.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  CV Macro F1   : {cv_f1:.4f} ± {cv_results['test_f1_macro'].std():.4f}")

    # -- Save --
    metrics = {
        "model":          model_name,
        "task":           "CrackWidth",
        "test_accuracy":  round(acc,      4),
        "test_f1_macro":  round(f1_macro, 4),
        "test_f1_02mm":   round(float(f1_per[0]), 4),
        "test_f1_04mm":   round(float(f1_per[1]), 4),
        "test_f1_06mm":   round(float(f1_per[2]), 4),
        "cv_accuracy":    round(cv_acc,   4),
        "cv_f1_macro":    round(cv_f1,    4),
    }
    save_metrics(metrics, output_dir)
    plot_confusion_matrix(y_test, preds, model_name, acc, f1_macro, output_dir)
    plot_cv_scores(cv_results, model_name, output_dir)

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("--- Loading crack width data ---")
    X, y, is_test = load_width_data(FEATURES_FILE)

    # ----------------------------------------------------------
    # 2. Specimen-level split  (Series 1 → train, Series 2 → test)
    # ----------------------------------------------------------
    X_train, y_train = X[~is_test], y[~is_test]
    X_test,  y_test  = X[is_test],  y[is_test]
    print(f"\nSpecimen-level split: train={len(y_train)} (series 1), test={len(y_test)} (series 2)")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Train {name}: {(y_train==i).sum()}  |  Test {name}: {(y_test==i).sum()}")

    # ----------------------------------------------------------
    # 3. Random Forest
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("Random Forest")
    print("=" * 50)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    evaluate_and_save(rf, X_train, y_train, X_test, y_test,
                      "RandomForest", os.path.join(RESULTS_ROOT, "RandomForest"))

    # ----------------------------------------------------------
    # 4. XGBoost
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("XGBoost")
    print("=" * 50)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    evaluate_and_save(xgb_clf, X_train, y_train, X_test, y_test,
                      "XGBoost", os.path.join(RESULTS_ROOT, "XGBoost"))

    # ----------------------------------------------------------
    # 5. SVM  (StandardScaler inside pipeline)
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("SVM")
    print("=" * 50)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)),
    ])
    evaluate_and_save(svm, X_train, y_train, X_test, y_test,
                      "SVM", os.path.join(RESULTS_ROOT, "SVM"))

    print(f"\nAll outputs saved under: {RESULTS_ROOT}/")


if __name__ == "__main__":
    main()
