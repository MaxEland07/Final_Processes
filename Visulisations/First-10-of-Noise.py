import numpy as np
import os
import matplotlib.pyplot as plt

# Define paths
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data")
noise_dir = os.path.join(base_dir, "Noise-Data")

# Parameters
record = "104"  # Record number to visualize
snr_level = 24  # SNR level in dB (e.g., 24 dB)
configs_to_plot = ["G", "BW_MA", "PL", "MA", "PL_G"]  # Example configs: Gaussian, Baseline Wander + Muscle Artifact

# Function to plot first 10 seconds with annotations
def plot_first_10_seconds_with_labels(signals, fs, sample_idx, labels, title, clean_signals=None):
    """
    Plot the first 10 seconds of both channels from an ECG signal with annotations below.
    Optionally overlay clean signals for comparison.

    Parameters:
    - signals: ndarray, shape (n_samples, 2), ECG signal with two channels.
    - fs: int, sampling frequency in Hz.
    - sample_idx: ndarray, annotation sample indices.
    - labels: ndarray, annotation labels corresponding to sample indices.
    - title: str, title for the plot.
    - clean_signals: ndarray, optional clean signal to overlay.
    """
    duration = 10  # seconds
    num_samples = int(duration * fs)  # Number of samples for 10 seconds

    # Create time vector
    time = np.linspace(0, duration, num_samples)

    # Extract data for the first 10 seconds
    channel_1 = signals[:num_samples, 0]
    channel_2 = signals[:num_samples, 1]
    clean_ch1 = clean_signals[:num_samples, 0] if clean_signals is not None else None
    clean_ch2 = clean_signals[:num_samples, 1] if clean_signals is not None else None

    # Filter annotations within the first 10 seconds
    annotations_in_range = [(idx, label) for idx, label in zip(sample_idx, labels) if idx < num_samples]

    # Plotting
    plt.figure(figsize=(12, 8))

    # Channel 1
    plt.subplot(2, 1, 1)
    plt.plot(time, channel_1, label="Signal (Channel 1)", color="blue")
    if clean_ch1 is not None:
        plt.plot(time, clean_ch1, label="Clean (Channel 1)", color="gray", alpha=0.5, linestyle="--")
    plt.title(f"{title} - Channel 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    for idx, label in annotations_in_range:
        plt.text(idx / fs, min(channel_1) - abs(min(channel_1)) * 0.1,
                 label, fontsize=8, color="red", ha="center")

    # Channel 2
    plt.subplot(2, 1, 2)
    plt.plot(time, channel_2, label="Signal (Channel 2)", color="green")
    if clean_ch2 is not None:
        plt.plot(time, clean_ch2, label="Clean (Channel 2)", color="gray", alpha=0.5, linestyle="--")
    plt.title(f"{title} - Channel 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    for idx, label in annotations_in_range:
        plt.text(idx / fs, min(channel_2) - abs(min(channel_2)) * 0.1,
                 label, fontsize=8, color="red", ha="center")

    plt.tight_layout()
    plt.show()

# Load clean signal
clean_file = f"{record}_full.npz"  # Assuming Processed-Data uses <record>.npz
clean_path = os.path.join(processed_dir, clean_file)
if os.path.exists(clean_path):
    clean_data = np.load(clean_path)
    clean_signals = clean_data["signals"]
    fs = int(clean_data["fs"])
    sample_idx = clean_data["sample_indices"]
    labels = clean_data["labels"]

    # Plot clean signal
    plot_first_10_seconds_with_labels(clean_signals, fs, sample_idx, labels, "Clean ECG")
else:
    print(f"Clean file not found: {clean_path}")
    clean_signals = None  # Fallback if clean data is unavailable

# Load and plot noisy samples
for config in configs_to_plot:
    # Construct noisy file path
    noisy_file = f"record_{record}_full_{snr_level}dB.npz"
    file_path = os.path.join(noise_dir, config, noisy_file)

    if os.path.exists(file_path):
        # Load noisy data
        data = np.load(file_path)
        noisy_signals = data["signals"]
        fs = int(data["fs"])
        sample_idx = data["sample_indices"]
        labels = data["labels"]

        # Plot noisy signal with clean overlay
        title = f"Noisy ECG (Config: {config}, SNR: {snr_level} dB)"
        plot_first_10_seconds_with_labels(noisy_signals, fs, sample_idx, labels, title, clean_signals)
    else:
        print(f"Noisy file not found: {file_path}")