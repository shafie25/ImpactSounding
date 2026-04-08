# ============================================================
# Dominant Frequency (Df) Feature Extraction Using FFT
# Acoustic Impact Hammer Testing
#
# This script:
# 1) Reads validated hammer-impact sound segments.
# 2) Converts each signal to the frequency domain using FFT.
# 3) Finds the frequency component with the maximum magnitude.
# 4) Saves the dominant frequency (Df) and its amplitude
#    for use in AI-based crack identification.
# ============================================================

import os
import glob
import librosa
import numpy as np
import pandas as pd


# ============================================================
# ----------------------- PARAMETERS -------------------------
# ============================================================

# Directory containing individual hammer-impact WAV files.
# These files are assumed to be already segmented,
# trimmed, and filtered for noise.
INPUT_SOUND_DIR = "Sounds"

# Output Excel file where the extracted Df features
# will be stored for later machine-learning analysis.
OUTPUT_EXCEL_FILE = "Dominant_Frequency_Df_Features.xlsx"


# ============================================================
# Dataset container
# ============================================================
#
# Each row in this list will store:
#   [Signal_ID, Dominant_Frequency_Hz, Dominant_Amplitude]
#
# Signal_ID:
#   Unique identifier for each hammer impact,
#   sensor position, or test location.
#
# Dominant_Frequency_Hz:
#   Frequency (in Hz) with the highest spectral magnitude.
#   This is defined as the Dominant Frequency (Df).
#
# Dominant_Amplitude:
#   Magnitude of the FFT at Df, representing
#   the strength of vibration at that frequency.
# ============================================================

dominant_frequency_dataset = []


# ============================================================
# -------------------- MAIN PROCESS LOOP --------------------
# ============================================================

for audio_file_path in glob.iglob(f"{INPUT_SOUND_DIR}/*.wav", recursive=True):

    # --------------------------------------------------------
    # Load acoustic impact signal
    # --------------------------------------------------------
    #
    # librosa.load() returns:
    #
    #   time_domain_signal :
    #       One-dimensional waveform.
    #
    #   sampling_rate :
    #       Sampling frequency in Hz.
    #
    # A consistent sampling rate ensures
    # correct frequency mapping.
    # --------------------------------------------------------

    time_domain_signal, sampling_rate = librosa.load(audio_file_path)

    # --------------------------------------------------------
    # Fast Fourier Transform (FFT)
    # --------------------------------------------------------
    #
    # FFT transforms the hammer response
    # from the time domain into the frequency domain.
    #
    # The output is complex-valued and contains
    # both amplitude and phase information.
    #
    # Only magnitude is retained because
    # Df is defined by energy content.
    # --------------------------------------------------------

    fft_complex_spectrum = np.fft.fft(time_domain_signal)

    fft_magnitude_spectrum = np.abs(fft_complex_spectrum)

    # --------------------------------------------------------
    # Frequency axis construction
    # --------------------------------------------------------
    #
    # Each FFT bin must be assigned
    # a physical frequency value (Hz).
    #
    # Frequencies span from:
    #   0 Hz → sampling_rate.
    #
    # Because the signal is real-valued,
    # the spectrum is symmetric around
    # the Nyquist frequency.
    #
    # Therefore, only the first half
    # of the spectrum is physically meaningful.
    # --------------------------------------------------------

    frequency_axis = np.linspace(
        0,
        sampling_rate,
        len(fft_magnitude_spectrum)
    )

    half_spectrum_index = int(len(fft_magnitude_spectrum) / 2)

    positive_frequencies = frequency_axis[:half_spectrum_index]
    positive_magnitudes = fft_magnitude_spectrum[:half_spectrum_index]

    # --------------------------------------------------------
    # Identify dominant frequency (Df)
    # --------------------------------------------------------
    #
    # np.argmax() finds the index of the maximum
    # magnitude in the positive-frequency spectrum.
    #
    # The corresponding frequency is defined
    # as the Dominant Frequency (Df).
    #
    # This feature is widely used in
    # acoustic-based damage detection
    # because cracking alters stiffness
    # and shifts resonance peaks.
    # --------------------------------------------------------

    dominant_index = np.argmax(positive_magnitudes)

    dominant_frequency_hz = float(
        positive_frequencies[dominant_index]
    )

    dominant_amplitude = float(
        positive_magnitudes[dominant_index]
    )

    # --------------------------------------------------------
    # Store extracted Df features
    # --------------------------------------------------------
    #
    # These scalar values are ready to be used
    # directly in machine-learning classifiers
    # or statistical analysis.
    # --------------------------------------------------------

    dominant_frequency_dataset.append([
        os.path.splitext(
            os.path.basename(audio_file_path)
        )[0],
        dominant_frequency_hz,
        dominant_amplitude
    ])


# ============================================================
# Convert results to DataFrame
# ============================================================
#
# A tabular format simplifies:
#   • dataset labeling
#   • classifier training
#   • statistical comparison
# ============================================================

df_dominant_frequency = pd.DataFrame(
    dominant_frequency_dataset,
    columns=[
        "Signal_ID",
        "Dominant_Frequency_Hz",
        "Dominant_Amplitude"
    ]
)


# ============================================================
# Save results to Excel file
# ============================================================
#
# The file name explicitly indicates:
#   - Dominant Frequency feature
#   - abbreviation Df
#   - FFT-based derivation
#
# This is appropriate for
# supplementary research data.
# ============================================================

df_dominant_frequency.to_excel(
    OUTPUT_EXCEL_FILE,
    index=False
)
