# ============================================================
# Classifier 3 — SVM (Support Vector Machine)
# Acoustic Impact Hammer Testing: Cracked vs Intact
#
# Loads Labeled_Features.xlsx, trains an SVM classifier,
# evaluates performance, and saves results + plots to
# Classifier Results/SVM/
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

FEATURES_FILE = "Labeled_Features.xlsx"
OUTPUT_DIR    = os.path.join("Classifier Results", "CrackDetection", "SVM")
RANDOM_STATE  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load data
# ============================================================

print("Loading features...")
df = pd.read_excel(FEATURES_FILE)

meta_cols    = ["Signal_ID", "Label", "Crack_Position", "Specimen"]
feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols].values
y = LabelEncoder().fit_transform(df["Label"])          # Cracked=0, Intact=1
class_names = ["Cracked", "Intact"]

print(f"  Samples: {len(y)}  |  Features: {len(feature_cols)}")
print(f"  Cracked: {(y==0).sum()}  |  Intact: {(y==1).sum()}")


# ============================================================
# Train / test split (stratified 80/20)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


# ============================================================
# Train SVM (inside a Pipeline with StandardScaler)
# SVM is sensitive to feature scale so scaling is required
# ============================================================

print("Training SVM (this may take a minute)...")

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(
        kernel="rbf",
        class_weight="balanced",   # compensates for class imbalance
        C=10,
        gamma="scale",
        random_state=RANDOM_STATE
    ))
])
clf.fit(X_train, y_train)
joblib.dump(clf, os.path.join(OUTPUT_DIR, "model.joblib"))
print("Saved: model.joblib")


# ============================================================
# Evaluate on test set
# ============================================================

y_pred = clf.predict(X_test)

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=0)
recall    = recall_score(y_test, y_pred, pos_label=0)
f1        = f1_score(y_test, y_pred, pos_label=0)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy : {acc:.4f}")
print(f"  Precision: {precision:.4f}  (Cracked)")
print(f"  Recall   : {recall:.4f}  (Cracked)")
print(f"  F1       : {f1:.4f}  (Cracked)")
print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")


# ============================================================
# 5-fold cross-validation
# ============================================================

print("Running 5-fold cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    clf, X, y, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"],
    n_jobs=-1
)

cv_summary = {k: float(np.mean(v)) for k, v in cv_results.items() if k.startswith("test_")}
print(f"  CV Accuracy : {cv_summary['test_accuracy']:.4f}")
print(f"  CV Precision: {cv_summary['test_precision']:.4f}")
print(f"  CV Recall   : {cv_summary['test_recall']:.4f}")
print(f"  CV F1       : {cv_summary['test_f1']:.4f}")


# ============================================================
# Save metrics to JSON (for comparison script)
# ============================================================

results = {
    "model":          "SVM",
    "test_accuracy":  acc,
    "test_precision": precision,
    "test_recall":    recall,
    "test_f1":        f1,
    "cv_accuracy":    cv_summary["test_accuracy"],
    "cv_precision":   cv_summary["test_precision"],
    "cv_recall":      cv_summary["test_recall"],
    "cv_f1":          cv_summary["test_f1"],
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)


# ============================================================
# Plot 1 — Confusion matrix
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("SVM — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")


# ============================================================
# Plot 2 — CV score distribution
# ============================================================

cv_metrics = ["accuracy", "precision", "recall", "f1"]
cv_means   = [cv_summary[f"test_{m}"] for m in cv_metrics]
cv_stds    = [float(np.std(cv_results[f"test_{m}"])) for m in cv_metrics]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(cv_metrics, cv_means, yerr=cv_stds, capsize=6,
              color=["steelblue", "darkorange", "seagreen", "crimson"], alpha=0.85)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("SVM — 5-Fold CV Scores (mean +/- std)")
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cv_scores.png"), dpi=150)
plt.close()
print("Saved: cv_scores.png")

print(f"\nAll results saved to: {OUTPUT_DIR}/")
