import os
import numpy as np
import matplotlib.pyplot as plt

# Define directories
base_dir = "/kaggle/working/MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data")
noisy_dir = os.path.join(processed_dir, "noisy_data")

# Function to view a random sample from a noisy file and compare with clean signal
def view_random_sample(file_name=None):
    # Default to the first noisy file if none specified
    if file_name is None:
        file_list = [f for f in os.listdir(noisy_dir) if f.endswith('.npz')]
        file_name = file_list[0]

    # Load the noisy data
    noisy_file_path = os.path.join(noisy_dir, file_name)
    noisy_data = np.load(noisy_file_path)
    noisy_signals = noisy_data['signals'][:, 0]  # Use only first channel
    fs = noisy_data['fs']

    # Load the clean data
    clean_file_path = os.path.join(processed_dir, "100_full.npz")
    clean_data = np.load(clean_file_path)
    clean_signals = clean_data['signals'][:, 0]  # Use only first channel

    # Determine signal length and select random starting point
    signal_length = min(len(noisy_signals), len(clean_signals))
    segment_length = 5000  # Number of samples to plot
    start_idx = np.random.randint(0, max(0, signal_length - segment_length))
    end_idx = start_idx + segment_length

    # Extract segments
    noisy_segment = noisy_signals[start_idx:end_idx]
    clean_segment = clean_signals[start_idx:end_idx]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(clean_segment, label='Clean Signal', color='black', linewidth=2)
    plt.plot(noisy_segment, label='Noisy Signal', color='red', alpha=0.7)
    plt.title(f"Comparison: Clean vs Noisy ({file_name}) at SNR = 6 dB\nStart Index: {start_idx}")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage: view a specific file or use default
    view_random_sample("100_full_em_snr6.npz")  # Specify a file, or omit to use default