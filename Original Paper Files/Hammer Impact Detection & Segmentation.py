# ============================================================
# Hammer Impact Sound Detection and Segmentation
# Acoustic Impact Hammer Testing Preprocessing
#
# This script:
# 1) Reads long acoustic recordings containing multiple hammer hits.
# 2) Splits them into individual impacts using silence detection.
# 3) Trims each impact to retain the direct-wave region.
# 4) Removes low-energy/noise segments.
# 5) Saves validated impacts and waveform figures.
# ============================================================

import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

from pydub import AudioSegment
from pydub.silence import split_on_silence


# ============================================================
# ----------------------- PARAMETERS -------------------------
# ============================================================

# Root folder containing all raw hammer-test recordings.
# Each WAV file may include multiple hammer impacts.
INPUT_ROOT = "ALL data"

# Root folder where all processed and segmented data will be saved.
OUTPUT_ROOT = "Split Data"

# Target sampling rate (Hz) used when re-loading and trimming segments.
# 22050 Hz is standard in audio processing and is sufficient
# to capture the frequency content of impact responses in concrete.
TARGET_SR = 22050

# Time offset (seconds) from the beginning of each detected segment
# before extracting the final signal window.
# This avoids including the silence padding kept during segmentation.
TRIM_OFFSET = 0.13

# Length (seconds) of the final extracted impact signal.
# This window is intended to capture the initial hammer strike
# and early vibration response, which dominate spectral features.
TRIM_DURATION = 0.20

# Minimum acceptable peak amplitude for a valid hammer impact.
# Segments with peak amplitude below this threshold are assumed
# to be background noise or weak/unintended taps and are discarded.
MIN_PEAK_AMPLITUDE = 0.30


# ============================================================
# -------------------- MAIN PROCESS LOOP --------------------
# ============================================================

for wav_path in glob.glob(f"{INPUT_ROOT}/**/*.wav", recursive=True):

    # --------------------------------------------------------
    # Extract signal identifier from file name
    # This typically represents the hammering location
    # or test position on the concrete specimen.
    # --------------------------------------------------------

    impact_position_id = os.path.splitext(
        os.path.basename(wav_path)
    )[0]

    # Extract the name of the parent directory.
    # This may correspond to specimen ID, test series,
    # or damage condition.
    parent_folder = os.path.basename(
        os.path.dirname(wav_path)
    )

    print(f"Processing: {impact_position_id}")

    # --------------------------------------------------------
    # Create structured output directories
    # --------------------------------------------------------
    #
    # For each impact position, three subfolders are created:
    #
    #   Sounds/       -> validated hammer impacts (WAV)
    #   Plots/        -> waveform visualizations (JPG)
    #   Spectrogram/ -> reserved for time-frequency plots
    #
    # This structure supports traceability and later analysis.
    # --------------------------------------------------------

    base_output_dir = os.path.join(
        OUTPUT_ROOT,
        parent_folder,
        impact_position_id
    )

    sound_dir = os.path.join(base_output_dir, "Sounds")
    plot_dir = os.path.join(base_output_dir, "Plots")
    spectrogram_dir = os.path.join(base_output_dir, "Spectrogram")

    for directory in [sound_dir, plot_dir, spectrogram_dir]:
        os.makedirs(directory, exist_ok=True)

    # --------------------------------------------------------
    # Load the full raw recording
    # --------------------------------------------------------
    #
    # The AudioSegment object keeps the original recording
    # including silent intervals between hammer strikes.
    # --------------------------------------------------------

    full_recording = AudioSegment.from_wav(wav_path)

    # --------------------------------------------------------
    # Silence-based segmentation
    # --------------------------------------------------------
    #
    # split_on_silence() separates hammer impacts by detecting
    # quiet regions between them.
    #
    # min_silence_len = 50 ms:
    #     Minimum duration that must be quiet before a new
    #     segment is created.
    #
    # silence_thresh = -50 dBFS:
    #     Sound level below which audio is considered silence.
    #     Values closer to 0 are stricter; more negative values
    #     allow quieter signals to be kept.
    #
    # keep_silence = 200 ms:
    #     Amount of audio preserved before and after each
    #     detected impact to avoid cutting off the onset.
    # --------------------------------------------------------

    audio_chunks = split_on_silence(
        full_recording,
        min_silence_len=50,
        silence_thresh=-50,
        keep_silence=200
    )

    # --------------------------------------------------------
    # Save all segmented chunks before validation
    # --------------------------------------------------------

    for i, chunk in enumerate(audio_chunks):

        output_chunk_path = os.path.join(
            sound_dir,
            f"{i:03d}.wav"
        )

        chunk.export(output_chunk_path, format="wav")

    # --------------------------------------------------------
    # Post-process each chunk:
    # trimming and quality control
    # --------------------------------------------------------

    for chunk_path in glob.glob(os.path.join(sound_dir, "*.wav")):

        chunk_id = os.path.splitext(
            os.path.basename(chunk_path)
        )[0]

        # ----------------------------------------------------
        # Load chunk and resample to TARGET_SR
        # ----------------------------------------------------
        #
        # offset:
        #     Time (seconds) skipped at the beginning.
        #     This removes the silence padding kept earlier.
        #
        # duration:
        #     Final length of the extracted impact signal.
        # ----------------------------------------------------

        signal, sr = librosa.load(
            chunk_path,
            sr=TARGET_SR,
            offset=TRIM_OFFSET,
            duration=TRIM_DURATION
        )

        # Save trimmed signal back to disk
        sf.write(chunk_path, signal, sr)

        # Reload trimmed signal for verification
        signal, sr = librosa.load(chunk_path, sr=TARGET_SR)

        # ----------------------------------------------------
        # Peak amplitude evaluation
        # ----------------------------------------------------
        #
        # peak_amplitude:
        #     Maximum absolute waveform value.
        #
        # This is used as a simple energy-based criterion
        # to reject background noise segments.
        # ----------------------------------------------------

        peak_amplitude = np.max(np.abs(signal))

        if peak_amplitude < MIN_PEAK_AMPLITUDE:

            # Remove segments that do not contain
            # a sufficiently strong hammer impact.
            os.remove(chunk_path)
            print(f"Removed (noise): {chunk_id}")
            continue

        # ----------------------------------------------------
        # Save waveform visualization
        # ----------------------------------------------------
        #
        # Figure size is selected to be compact and
        # suitable for reporting.
        #
        # dpi = 200:
        #     High enough resolution for publication figures.
        # ----------------------------------------------------

        plt.figure(figsize=(6, 3))
        librosa.display.waveshow(signal, sr=sr)
        plt.title(chunk_id)

        plot_path = os.path.join(
            plot_dir,
            f"{chunk_id}.jpg"
        )

        plt.savefig(plot_path, dpi=200)
        plt.close()
