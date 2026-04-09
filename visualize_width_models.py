# ============================================================
# visualize_width_models.py
# Model visualization for CrackWidth CNN_1D and MLP
#
# Produces (saved to Classifier Results/CrackWidth/Visualization/):
#   1. torchinfo architecture summaries (.txt)
#   2. ONNX exports — open in https://netron.app
#   3. Grad-CAM heatmaps for CNN_1D — one set per width class
#      (which waveform regions drive each width prediction)
#
# Usage:
#   python visualize_width_models.py
# ============================================================

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from torchinfo import summary


# ============================================================
# PATHS
# ============================================================

CNN1D_PT   = os.path.join("Classifier Results", "CrackWidth", "CNN_1D", "best_model.pt")
MLP_PT     = os.path.join("Classifier Results", "CrackWidth", "MLP",    "best_model.pt")
DATA_DIR   = os.path.join("Specimens", "Class_sounds", "Cracked")
OUTPUT_DIR = os.path.join("Classifier Results", "CrackWidth", "Visualization")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE  = 22050
NUM_SAMPLES  = 4410
GRADCAM_N    = 3          # samples per class
CLASS_NAMES  = ["0.2mm", "0.4mm", "0.6mm"]
NUM_CLASSES  = 3


# ============================================================
# MODEL DEFINITIONS  (must match training scripts exactly)
# ============================================================

class CNN1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
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


class MLP(nn.Module):
    def __init__(self, input_dim=29, num_classes=NUM_CLASSES, dropout=0.3):
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
# HELPERS
# ============================================================

def save_summary(model, input_size, name):
    path = os.path.join(OUTPUT_DIR, f"{name}_summary.txt")
    s = summary(model, input_size=input_size, verbose=0,
                col_names=["input_size", "output_size", "num_params", "trainable"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(s))
    print(f"  Saved: {path}")


def export_onnx(model, dummy_input, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    torch.onnx.export(
        model, dummy_input, path,
        input_names=["input"], output_names=["logits"],
        opset_version=17, verbose=False,
    )
    print(f"  Saved: {path}  (drag into https://netron.app)")


def load_wav(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,4410)


def collect_wavs_by_width(data_dir, n):
    """
    Return {label_idx: [wav_path, ...]} for up to n samples per class.
    Width is derived from folder name: specimen_X_Y → Y-1 = label.
    """
    pattern = re.compile(r"specimen_\d+_(\d+)")
    samples = {0: [], 1: [], 2: []}
    for folder in sorted(os.listdir(data_dir)):
        match = pattern.search(folder)
        if not match:
            continue
        label = int(match.group(1)) - 1
        if len(samples[label]) >= n:
            continue
        sounds_dir = os.path.join(data_dir, folder, "Sounds")
        if not os.path.isdir(sounds_dir):
            sounds_dir = os.path.join(data_dir, folder)
        for fname in sorted(os.listdir(sounds_dir)):
            if fname.lower().endswith(".wav") and len(samples[label]) < n:
                samples[label].append(os.path.join(sounds_dir, fname))
    return samples


# ============================================================
# GRAD-CAM (same implementation as visualize_models.py)
# ============================================================

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients[0].mean(dim=-1)
        cam = (weights[:, None] * self.activations[0]).sum(dim=0)
        cam = torch.clamp(cam, min=0).numpy().astype(np.float32)

        cam_up = np.interp(
            np.linspace(0, 1, NUM_SAMPLES),
            np.linspace(0, 1, len(cam)),
            cam,
        )
        if cam_up.max() > 0:
            cam_up /= cam_up.max()
        return cam_up


def plot_gradcam(waveform_np, cam, true_class, pred_class, save_path):
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE * 1000

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    points = np.array([t, waveform_np]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    norm   = Normalize(vmin=0, vmax=1)
    lc     = LineCollection(segs, cmap="jet", norm=norm, linewidth=1.2)
    lc.set_array(cam[:-1])
    ax.add_collection(lc)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(waveform_np.min() * 1.15, waveform_np.max() * 1.15)
    ax.set_ylabel("Amplitude")
    correct = "[correct]" if true_class == pred_class else "[wrong]"
    ax.set_title(
        f"Grad-CAM — True: {CLASS_NAMES[true_class]}  |  "
        f"Predicted: {CLASS_NAMES[pred_class]}  {correct}",
        fontsize=11,
    )
    fig.colorbar(lc, ax=ax, orientation="vertical", pad=0.01).set_label("CAM intensity", fontsize=9)

    axes[1].fill_between(t, cam, alpha=0.7, color="crimson")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cpu")

    # ----------------------------------------------------------
    # 1. CNN_1D (Width) — summary + ONNX + Grad-CAM
    # ----------------------------------------------------------
    print("=" * 60)
    print("CNN_1D  (raw waveform -> 3-class width)")
    print("=" * 60)

    cnn1d = CNN1D(num_classes=NUM_CLASSES).to(device)
    cnn1d.load_state_dict(torch.load(CNN1D_PT, map_location=device))
    cnn1d.eval()

    print("\n-- Architecture summary --")
    save_summary(cnn1d, input_size=(1, 1, NUM_SAMPLES), name="cnn1d_width")

    print("-- ONNX export --")
    export_onnx(cnn1d, torch.zeros(1, 1, NUM_SAMPLES), "cnn1d_width")

    print(f"-- Grad-CAM ({GRADCAM_N} samples per class) --")
    grad_cam = GradCAM1D(cnn1d, cnn1d.conv_blocks[11])
    wav_samples = collect_wavs_by_width(DATA_DIR, GRADCAM_N)

    for label_idx, class_name in enumerate(CLASS_NAMES):
        for i, wav_path in enumerate(wav_samples[label_idx]):
            x = load_wav(wav_path).to(device)
            cam = grad_cam(x, class_idx=label_idx)
            pred_idx = cnn1d(x).argmax(dim=1).item()
            waveform_np = x.squeeze().detach().numpy()
            fname = f"gradcam_{class_name}_sample{i+1}.png"
            plot_gradcam(waveform_np, cam, label_idx, pred_idx,
                         os.path.join(OUTPUT_DIR, fname))

    # ----------------------------------------------------------
    # 2. MLP (Width) — summary + ONNX
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("MLP  (29 tabular features -> 3-class width)")
    print("=" * 60)

    mlp = MLP(input_dim=29, num_classes=NUM_CLASSES).to(device)
    mlp.load_state_dict(torch.load(MLP_PT, map_location=device))
    mlp.eval()

    print("\n-- Architecture summary --")
    save_summary(mlp, input_size=(1, 29), name="mlp_width")

    print("-- ONNX export --")
    export_onnx(mlp, torch.zeros(1, 29), "mlp_width")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")
    print("\nTo view ONNX graphs:")
    print("  1. Go to https://netron.app")
    print("  2. Drag in cnn1d_width.onnx or mlp_width.onnx")


if __name__ == "__main__":
    main()
