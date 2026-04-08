# ============================================================
# MFCC Image Generator
# Acoustic Impact Hammer Testing
#
# This script:
# 1) Walks Specimens/Class_sounds/Cracked/ and Intact/
# 2) Computes MFCC spectrograms for every WAV file using
#    the same parameters as Labeled Feature Extraction.py
#    (n_mfcc=13, sr=None, default librosa hop/fft)
# 3) Saves each spectrogram as a 224x224 JPG to:
#      MFCC_Images/Cracked/
#      MFCC_Images/Intact/
#    ready for ImageFolder-based CNN training
# ============================================================

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# PARAMETERS — kept identical to Labeled Feature Extraction.py
# ============================================================

CLASS_SOUNDS_DIR = os.path.join("Specimens", "Class_sounds")
OUTPUT_DIR       = "MFCC_Images"

NUM_MFCC         = 13          # all 13 coefficients (our standard)
IMAGE_SIZE_PX    = 224         # 224x224 — standard CNN input size
IMAGE_DPI        = 100         # figsize = IMAGE_SIZE_PX / IMAGE_DPI


# ============================================================
# HELPER — build a unique flat filename from the WAV path
# e.g. Cracked: specimen_1_1SoundDataC02_000.jpg
#      Intact:  specimen_1_1_specimen_1_1SoundDataN02_000.jpg
# ============================================================

def make_image_name(wav_path, label_dir):
    rel   = os.path.relpath(wav_path, label_dir)
    parts = rel.replace("\\", "/").split("/")
    parts = [p for p in parts if p != "Sounds"]     # drop the "Sounds" folder
    name  = "_".join(parts).replace(".wav", ".jpg")
    return name


# ============================================================
# HELPER — generate and save one MFCC spectrogram image
# ============================================================

def save_mfcc_image(wav_path, output_path):

    # Load at native sample rate — same as feature extraction
    signal, sr = librosa.load(wav_path, sr=None, mono=True)

    # Compute MFCC matrix — same params as Labeled Feature Extraction.py
    mfcc_matrix = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)

    # Build a figure that maps exactly to IMAGE_SIZE_PX x IMAGE_SIZE_PX
    fig_size = IMAGE_SIZE_PX / IMAGE_DPI
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=IMAGE_DPI)

    # Axis fills the entire figure — no margins, no whitespace
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Draw the MFCC spectrogram
    librosa.display.specshow(mfcc_matrix, sr=sr, ax=ax)

    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(fig)


# ============================================================
# MAIN — walk Cracked and Intact, generate all images
# ============================================================

total_saved   = 0
total_skipped = 0

for label in ["Cracked", "Intact"]:

    label_dir  = os.path.join(CLASS_SOUNDS_DIR, label)
    output_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(
        os.path.join(label_dir, "**", "*.wav"),
        recursive=True
    )

    print(f"\n{label}: {len(wav_files)} WAV files found")

    for wav_path in tqdm(wav_files, desc=label, unit="file"):

        image_name   = make_image_name(wav_path, label_dir)
        output_path  = os.path.join(output_dir, image_name)

        # Skip if already generated (allows resuming interrupted runs)
        if os.path.exists(output_path):
            total_skipped += 1
            continue

        try:
            save_mfcc_image(wav_path, output_path)
            total_saved += 1
        except Exception as e:
            print(f"\n  Skipping {os.path.basename(wav_path)}: {e}")


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*50}")
print(f"Done.")
print(f"  Images saved:   {total_saved}")
print(f"  Images skipped: {total_skipped} (already existed)")

for label in ["Cracked", "Intact"]:
    count = len(glob.glob(os.path.join(OUTPUT_DIR, label, "*.jpg")))
    print(f"  {label}: {count} images in {os.path.join(OUTPUT_DIR, label)}/")

print(f"{'='*50}")
