# ============================================================
# Frequency Feature Value (Vf) Extraction Using FFT
# Acoustic Impact Hammer Testing
#
# Correct definition:
# Source: https://www.robot.t.u-tokyo.ac.jp/~yamashita/paper/B/B021Final.pdf Sec. 2.2.2 Processing in frequency domain
#
#   As^2  = sum(A_i^2)                           → Total spectral energy
#   As    = sqrt(sum(A_i^2))                     → Energy normalization factor
#   f_bar = sum(A_i^2 * f_i) / sum(A_i^2)        → Energy-weighted mean frequency
#   Vf    = (1 / As) * sqrt( sum( A_i^2 (f_i - f_bar)^2 ) )
#           → Amplitude-weighted frequency spread
# ============================================================


# ------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------

import os              # For file and path operations
import glob            # For reading multiple files using wildcards
import librosa         # For audio loading and processing
import numpy as np     # For numerical and FFT operations
import pandas as pd    # For saving results to Excel


# ------------------------------------------------------------
# Define input/output locations
# ------------------------------------------------------------

INPUT_SOUND_DIR = "Sounds"                      # Folder containing .wav files
OUTPUT_EXCEL_FILE = "FFT_Feature_Value_Vf.xlsx" # Output Excel file name


# ------------------------------------------------------------
# Create container to store results
# ------------------------------------------------------------

vf_dataset = []   # Will store: [signal_id, Vf]


# ------------------------------------------------------------
# Loop through all WAV files inside the folder
# ------------------------------------------------------------

for audio_path in glob.iglob(f"{INPUT_SOUND_DIR}/*.wav", recursive=True):

    # Extract file name without extension (used as signal ID)
    signal_id = os.path.splitext(os.path.basename(audio_path))[0]
    print(f"Processing signal: {signal_id}")

    # --------------------------------------------------------
    # Load signal
    # --------------------------------------------------------
    # signal_time_series → audio samples in time domain
    # sampling_rate      → samples per second (Hz)
    signal_time_series, sampling_rate = librosa.load(
        audio_path,
        sr=None,     # Keep original sampling rate
        mono=True    # Convert to mono if stereo
    )

    # --------------------------------------------------------
    # Apply FFT (convert from time domain to frequency domain)
    # --------------------------------------------------------
    # rfft is used because audio signals are real-valued
    fft_spectrum = np.fft.rfft(signal_time_series)

    # Magnitude spectrum → A_i
    fft_magnitude = np.abs(fft_spectrum)

    # --------------------------------------------------------
    # Generate frequency axis (f_i values)
    # --------------------------------------------------------
    # Gives frequency value for each FFT bin
    frequency_axis = np.fft.rfftfreq(
        len(signal_time_series),
        d=1.0 / sampling_rate
    )

    # --------------------------------------------------------
    # Compute A_i^2 (power spectrum)
    # --------------------------------------------------------
    amplitude_squared = fft_magnitude ** 2

    # --------------------------------------------------------
    # As^2 = sum(A_i^2)
    # Total spectral energy
    # --------------------------------------------------------
    As_squared = np.sum(amplitude_squared)

    # Avoid division by zero in case of silent signal
    if As_squared == 0:
        print("Warning: Zero spectrum energy detected.")
        continue

    # --------------------------------------------------------
    # As = sqrt(sum(A_i^2))
    # Normalization term
    # --------------------------------------------------------
    As = np.sqrt(As_squared)

    # --------------------------------------------------------
    # Weighted mean frequency (f_bar)
    # f_bar = sum(A_i^2 * f_i) / sum(A_i^2)
    # --------------------------------------------------------
    f_bar = np.sum(amplitude_squared * frequency_axis) / As_squared

    # --------------------------------------------------------
    # Compute Vf (frequency feature value)
    # Vf = sqrt( sum( A_i^2 (f_i - f_bar)^2 ) ) / As
    # This measures frequency spread around the mean frequency
    # --------------------------------------------------------
    Vf = np.sqrt(
        np.sum(amplitude_squared * (frequency_axis - f_bar) ** 2)
    ) / As

    # Store result for this signal
    vf_dataset.append([signal_id, Vf])


# ------------------------------------------------------------
# Save results to Excel
# ------------------------------------------------------------

df_vf = pd.DataFrame(vf_dataset, columns=["Signal_ID", "Vf"])

# Write to Excel file
df_vf.to_excel(OUTPUT_EXCEL_FILE, index=False)

print(f"\nVf feature file saved to: {OUTPUT_EXCEL_FILE}")