# ============================================================
# visualize_models.py
# Model visualization for CNN_1D and CNN_MFCC
#
# Produces (all saved to Classifier Results/Visualization/):
#   1. torchinfo architecture summaries (.txt)
#   2. ONNX exports — open in https://netron.app for a graph
#   3. Grad-CAM heatmaps for CNN_1D (which waveform regions
#      drive each prediction)
#
# Usage:
#   python visualize_models.py
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchinfo import summary


# ============================================================
# PATHS
# ============================================================

CNN1D_PT    = os.path.join("Classifier Results", "CNN_1D",   "best_model.pt")
MFCC_PT     = os.path.join("Classifier Results", "CNN_MFCC", "best_head.pt")
DATA_DIR    = os.path.join("Specimens", "Class_sounds")
OUTPUT_DIR  = os.path.join("Classifier Results", "Visualization")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE  = 22050
NUM_SAMPLES  = 4410      # 0.20 s
GRADCAM_N    = 3         # number of samples per class for Grad-CAM


# ============================================================
# MODEL DEFINITIONS  (must match the training scripts exactly)
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
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_mfcc_head():
    """Classifier head trained on top of frozen EfficientNet-B0 features."""
    return nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )


# ============================================================
# HELPERS
# ============================================================

CLASS_NAMES = ["Cracked", "Intact"]


def load_wav(path):
    """Load a WAV and return a (1, 1, NUM_SAMPLES) float32 tensor."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,4410)


def collect_wavs(label_name, n):
    """Walk Class_sounds/<label_name>/ and return up to n WAV paths."""
    paths = []
    folder = os.path.join(DATA_DIR, label_name)
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                paths.append(os.path.join(root, f))
                if len(paths) >= n:
                    return paths
    return paths


def save_summary(model, input_size, name):
    """Write a torchinfo summary to a .txt file and print it."""
    path = os.path.join(OUTPUT_DIR, f"{name}_summary.txt")
    s = summary(model, input_size=input_size, verbose=0,
                col_names=["input_size", "output_size", "num_params", "trainable"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(s))
    print(f"  Saved: {path}\n")


def export_onnx(model, dummy_input, name):
    """Export model to ONNX (open the file at https://netron.app)."""
    path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    torch.onnx.export(
        model, dummy_input, path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        verbose=False,
    )
    print(f"  Saved:ONNX saved: {path}  (drag into https://netron.app to inspect)")


# ============================================================
# GRAD-CAM for CNN_1D
# Hooks the last ReLU in conv_blocks, computes class-weighted
# activation maps, and upsampls back to the raw waveform length.
# ============================================================

class GradCAM1D:
    """
    Grad-CAM for a 1D CNN.
    target_layer: the nn.Module to hook (we use the last ReLU in conv_blocks).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients   = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()   # (B, C, T)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()   # (B, C, T)

    def __call__(self, x, class_idx=None):
        """
        Run a forward+backward pass and return the CAM heatmap
        upsampled to input length.

        x          : (1, 1, NUM_SAMPLES) input tensor
        class_idx  : target class (None → use predicted class)
        Returns    : (NUM_SAMPLES,) numpy array, values in [0, 1]
        """
        self.model.eval()
        x = x.clone().requires_grad_(True)

        logits = self.model(x)               # (1, 2)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool the gradients over the time axis → (C,)
        weights = self.gradients[0].mean(dim=-1)          # (256,)
        # Weighted combination of activation maps → (T,)
        cam = (weights[:, None] * self.activations[0]).sum(dim=0)   # (T,)
        cam = torch.clamp(cam, min=0)                    # ReLU

        # Upsample to original input length
        cam = cam.numpy().astype(np.float32)
        cam_upsampled = np.interp(
            np.linspace(0, 1, NUM_SAMPLES),
            np.linspace(0, 1, len(cam)),
            cam,
        )
        # Normalise to [0, 1]
        if cam_upsampled.max() > 0:
            cam_upsampled /= cam_upsampled.max()

        return cam_upsampled, class_idx


def plot_gradcam(waveform_np, cam, true_label, pred_label, save_path):
    """
    Overlay Grad-CAM heat on the raw waveform.
    waveform_np : (NUM_SAMPLES,) float array
    cam         : (NUM_SAMPLES,) float array in [0,1]
    """
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE * 1000   # ms

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # --top panel: waveform coloured by CAM intensity --────────
    ax = axes[0]
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    points  = np.array([t, waveform_np]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    norm    = Normalize(vmin=0, vmax=1)
    lc      = LineCollection(segs, cmap="jet", norm=norm, linewidth=1.2)
    lc.set_array(cam[:-1])
    ax.add_collection(lc)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(waveform_np.min() * 1.15, waveform_np.max() * 1.15)
    ax.set_ylabel("Amplitude")
    correct = "[correct]" if true_label == pred_label else "[wrong]"
    ax.set_title(
        f"Grad-CAM — True: {CLASS_NAMES[true_label]}  |  "
        f"Predicted: {CLASS_NAMES[pred_label]}  {correct}",
        fontsize=11,
    )
    cbar = fig.colorbar(lc, ax=ax, orientation="vertical", pad=0.01)
    cbar.set_label("CAM intensity", fontsize=9)

    # --bottom panel: CAM as a filled area --───────────────────
    axes[1].fill_between(t, cam, alpha=0.7, color="crimson")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved:Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cpu")

    # ----------------------------------------------------------
    # 1.  CNN_1D  — summary + ONNX
    # ----------------------------------------------------------
    print("=" * 60)
    print("CNN_1D  (raw waveform -> 4-block Conv1D -> MLP)")
    print("=" * 60)

    cnn1d = CNN1D().to(device)
    cnn1d.load_state_dict(torch.load(CNN1D_PT, map_location=device))
    cnn1d.eval()

    print("\n--Architecture summary --")
    save_summary(cnn1d, input_size=(1, 1, NUM_SAMPLES), name="cnn1d")

    print("--ONNX export --")
    dummy_wave = torch.zeros(1, 1, NUM_SAMPLES)
    export_onnx(cnn1d, dummy_wave, "cnn1d")

    # ----------------------------------------------------------
    # 2.  CNN_MFCC head  — summary + ONNX
    # ----------------------------------------------------------
    print("=" * 60)
    print("CNN_MFCC head  (EfficientNet-B0 features -> MLP)")
    print("=" * 60)

    head = build_mfcc_head().to(device)
    head.load_state_dict(torch.load(MFCC_PT, map_location=device))
    head.eval()

    print("\n--Architecture summary --")
    save_summary(head, input_size=(1, 1280), name="cnn_mfcc_head")

    print("--ONNX export --")
    dummy_feat = torch.zeros(1, 1280)
    export_onnx(head, dummy_feat, "cnn_mfcc_head")

    # ----------------------------------------------------------
    # 3.  Grad-CAM on CNN_1D
    # ----------------------------------------------------------
    print("=" * 60)
    print(f"Grad-CAM  (CNN_1D, {GRADCAM_N} samples per class)")
    print("=" * 60)

    # Hook the last ReLU in conv_blocks (index 11)
    grad_cam = GradCAM1D(cnn1d, cnn1d.conv_blocks[11])

    for label_idx, label_name in enumerate(CLASS_NAMES):
        wav_paths = collect_wavs(label_name, GRADCAM_N)
        for i, wav_path in enumerate(wav_paths):
            x = load_wav(wav_path).to(device)           # (1, 1, 4410)
            cam, pred_idx = grad_cam(x, class_idx=label_idx)

            waveform_np = x.squeeze().numpy()
            fname = f"gradcam_{label_name.lower()}_sample{i+1}.png"
            plot_gradcam(
                waveform_np, cam,
                true_label=label_idx,
                pred_label=pred_idx,
                save_path=os.path.join(OUTPUT_DIR, fname),
            )

    print("\nDone. All outputs in:", OUTPUT_DIR)
    print("\nTo view the ONNX graphs:")
    print("  1. Go to https://netron.app")
    print("  2. Drag in cnn1d.onnx  or  cnn_mfcc_head.onnx")


if __name__ == "__main__":
    main()
