# ============================================================
# MFCC Feature Extraction & Visualization
# Acoustic Impact Hammer Testing
#
# This script:
# 1) Reads all WAV files inside "Sounds" folder
# 2) Computes MFCC features
# 3) Saves MFCC images into Sounds/MFCCs
# ============================================================


# ------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------

import os                       # File and directory handling
import glob                     # Reading multiple files
import librosa                  # Audio processing
import librosa.display          # MFCC visualization
import matplotlib.pyplot as plt # Plotting


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

# Folder containing all hammering sounds
INPUT_FOLDER = "Sounds"

# Folder to save MFCC images
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "MFCCs")

# Create output folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Number of MFCC coefficients (standard choice)
NUM_MFCC = 13

# Image resolution (DPI)
MFCC_IMAGE_DPI = 96

# Desired image size in pixels
MFCC_WIDTH_PX = 194
MFCC_HEIGHT_PX = 195


# ------------------------------------------------------------
# LOOP THROUGH ALL WAV FILES IN "Sounds"
# ------------------------------------------------------------

for audio_path in glob.glob(os.path.join(INPUT_FOLDER, "*.wav")):

    # Extract file name (without extension)
    file_name = os.path.splitext(os.path.basename(audio_path))[0]

    print(f"Processing file: {file_name}")

    # --------------------------------------------------------
    # Load audio signal
    # --------------------------------------------------------
    # signal → time-domain waveform
    # sr     → sampling rate
    signal, sr = librosa.load(audio_path, sr=None, mono=True)

    # --------------------------------------------------------
    # Compute MFCC matrix
    # --------------------------------------------------------
    # MFCC compresses spectral envelope information
    mfcc_matrix = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=NUM_MFCC
    )

    # --------------------------------------------------------
    # Create fixed-size figure
    # --------------------------------------------------------
    plt.figure(
        figsize=(
            MFCC_WIDTH_PX / MFCC_IMAGE_DPI,
            MFCC_HEIGHT_PX / MFCC_IMAGE_DPI
        ),
        frameon=False,
        dpi=MFCC_IMAGE_DPI
    )

    # Remove axes for CNN-ready clean image
    plt.axis("off")

    # Display MFCC as image
    librosa.display.specshow(mfcc_matrix, sr=sr)

    # --------------------------------------------------------
    # Save MFCC image
    # --------------------------------------------------------
    output_path = os.path.join(OUTPUT_FOLDER, file_name + ".jpg")

    plt.savefig(
        output_path,
        bbox_inches="tight",  # Remove margins
        pad_inches=0,         # No padding
        dpi=MFCC_IMAGE_DPI
    )

    # Free memory
    plt.close()



print("\nAll MFCC images successfully generated.")