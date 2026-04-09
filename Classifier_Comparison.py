# ============================================================
# Classifier Comparison
# Acoustic Impact Hammer Testing: Cracked vs Intact
#
# Loads metrics.json from each classifier's output folder
# and produces side-by-side comparison plots.
#
# Run AFTER all four classifier scripts have completed.
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RESULTS_ROOT = os.path.join("Classifier Results", "CrackDetection")
CLASSIFIERS  = ["RandomForest", "XGBoost", "SVM", "CNN_MFCC", "CNN_1D", "LSTM", "MLP"]
OUTPUT_DIR   = os.path.join(RESULTS_ROOT, "Comparison")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load metrics from each classifier
# ============================================================

records = []
for clf_name in CLASSIFIERS:
    metrics_path = os.path.join(RESULTS_ROOT, clf_name, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"WARNING: {metrics_path} not found — run {clf_name} script first.")
        continue
    with open(metrics_path) as f:
        records.append(json.load(f))

if not records:
    raise RuntimeError("No metrics found. Run the classifier scripts first.")

df = pd.DataFrame(records).set_index("model")
print("Loaded metrics:")
print(df.to_string())


# ============================================================
# Plot 1 — Test set metrics bar chart (grouped)
# ============================================================

test_metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
metric_labels = ["Accuracy", "Precision\n(Cracked)", "Recall\n(Cracked)", "F1\n(Cracked)"]

n_clf  = len(df)
x      = np.arange(len(metric_labels))
width  = 0.8 / n_clf
colors = ["steelblue", "darkorange", "seagreen", "crimson", "mediumpurple", "saddlebrown", "teal"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (clf_name, color) in enumerate(zip(df.index, colors)):
    vals   = [df.loc[clf_name, m] for m in test_metrics]
    offset = (i - (n_clf - 1) / 2) * width
    bars   = ax.bar(x + offset, vals, width, label=clf_name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Classifier Comparison — Test Set Metrics")
ax.legend()
ax.axhline(0.9, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_test_metrics_comparison.png"), dpi=150)
plt.close()
print("Saved: 1_test_metrics_comparison.png")


# ============================================================
# Plot 2 — CV metrics bar chart (grouped)
# ============================================================

cv_metrics    = ["cv_accuracy", "cv_precision", "cv_recall", "cv_f1"]
cv_labels     = ["CV Accuracy", "CV Precision\n(Cracked)", "CV Recall\n(Cracked)", "CV F1\n(Cracked)"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (clf_name, color) in enumerate(zip(df.index, colors)):
    vals   = [df.loc[clf_name, m] for m in cv_metrics]
    offset = (i - (n_clf - 1) / 2) * width
    bars   = ax.bar(x + offset, vals, width, label=clf_name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(cv_labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Classifier Comparison — CV Metrics\n(RF/XGBoost/SVM: 5-Fold CV  |  CNN_MFCC/CNN_1D: Best Validation)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_cv_metrics_comparison.png"), dpi=150)
plt.close()
print("Saved: 2_cv_metrics_comparison.png")


# ============================================================
# Plot 3 — Heatmap of all metrics
# ============================================================

all_metrics = test_metrics + cv_metrics
all_labels  = ["Test Acc", "Test Prec", "Test Rec", "Test F1",
               "CV Acc",   "CV Prec",   "CV Rec",   "CV F1"]

heatmap_data = df[all_metrics].copy()
heatmap_data.columns = all_labels

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGn",
            vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5)
ax.set_title("Classifier Comparison — All Metrics Heatmap\n(CNN_MFCC/CNN_1D cv_* = best validation metrics, not 5-fold CV)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_metrics_heatmap.png"), dpi=150)
plt.close()
print("Saved: 3_metrics_heatmap.png")


# ============================================================
# Plot 4 — Recall vs Precision scatter (test set)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))
for clf_name, color in zip(df.index, colors):
    ax.scatter(df.loc[clf_name, "test_precision"],
               df.loc[clf_name, "test_recall"],
               color=color, s=180, zorder=5, label=clf_name)
    ax.annotate(clf_name,
                (df.loc[clf_name, "test_precision"], df.loc[clf_name, "test_recall"]),
                textcoords="offset points", xytext=(8, 4), fontsize=10)

ax.set_xlabel("Precision (Cracked)")
ax.set_ylabel("Recall (Cracked)")
ax.set_title("Precision vs Recall — Cracked Class")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_precision_recall_scatter.png"), dpi=150)
plt.close()
print("Saved: 4_precision_recall_scatter.png")


# ============================================================
# Summary table
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY — Best classifier per metric")
print("=" * 60)
for m, label in zip(test_metrics, metric_labels):
    best = df[m].idxmax()
    print(f"  {label.replace(chr(10), ' '):<28} -> {best} ({df.loc[best, m]:.4f})")

print(f"\nAll comparison plots saved to: {OUTPUT_DIR}/")
