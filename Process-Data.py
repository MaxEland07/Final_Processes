import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# Define directories
base_dir = "./MIT-BIH Arrhythmia Database"
raw_dir = os.path.join(base_dir, "Raw-Data")
processed_dir = os.path.join(base_dir, "Processed-Data")

# Define filter design functions
def butter_highpass(cutoff, fs, order=5):
    """Design a high-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    """Design a low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, b, a):
    """Apply the designed filter to the data."""
    return filtfilt(b, a, data, axis=0)

def preprocess_data():
    print("Preprocessing MIT-BIH data into full record...")

    # Create the processed_dir if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created directory: {processed_dir}")

    # Use only record "100" (you can change this to any specific record)
    record = "100"
    record_path = os.path.join(raw_dir, record)

    # Delete existing files in the processed directory
    for file in os.listdir(processed_dir):
        file_path = os.path.join(processed_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    try:
        # Load the specific record
        data = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')

        signals = data.p_signal
        fs = data.fs
        sample_indices = ann.sample
        labels = ann.symbol

        # Design filters
        highpass_cutoff = 0.5  # Hz (to remove baseline wander)
        lowpass_cutoff = 50   # Hz (to remove high-frequency noise)
        hp_b, hp_a = butter_highpass(highpass_cutoff, fs)
        lp_b, lp_a = butter_lowpass(lowpass_cutoff, fs)

        # Apply high-pass filter (baseline wander removal)
        filtered_signals = apply_filter(signals, hp_b, hp_a)

        # Apply low-pass filter (high-frequency noise removal)
        filtered_signals = apply_filter(filtered_signals, lp_b, lp_a)

        # Save preprocessed signals to file
        output_file = os.path.join(processed_dir, f"{record}_full.npz")
        np.savez(output_file,
                 signals=filtered_signals,
                 sample_indices=sample_indices,
                 labels=labels,
                 fs=fs)

        print(f"Processed {record}: Full record saved to {output_file} (shape: {filtered_signals.shape})")

    except Exception as e:
        print(f"Error processing {record}: {e}")

if __name__ == "__main__":
    preprocess_data()
    print(f"Preprocessing Complete! Files saved in {processed_dir}")