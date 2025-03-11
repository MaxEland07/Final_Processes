import numpy as np
import os
import matplotlib.pyplot as plt

# Define paths
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data")

# Load the record
record = 100  # Change this to the desired record number
data = np.load(f'{processed_dir}/{record}_full.npz')

# Extract data
signals = data['signals']  # Shape: (65000, 2)
fs = int(data['fs'])  # Sampling frequency (e.g., 360 Hz)
sample_idx = data['sample_indices']  # Annotation sample indices
labels = data['labels']  # Annotation labels

# Plot the first 10 seconds of both channels with labels
def plot_first_10_seconds_with_labels(signals, fs, sample_idx, labels):
    """
    Plot the first 10 seconds of both channels from an ECG signal and display labels below each graph.

    Parameters:
    - signals: ndarray, shape (n_samples, 2), ECG signal with two channels.
    - fs: int, sampling frequency in Hz.
    - sample_idx: ndarray, annotation sample indices.
    - labels: ndarray, annotation labels corresponding to sample indices.
    """
    duration = 10  # seconds
    num_samples = int(duration * fs)  # Number of samples for the first 10 seconds

    # Create time vector
    time = np.linspace(0, duration, num_samples)

    # Extract data for the first 10 seconds
    channel_1 = signals[:num_samples, 0]
    channel_2 = signals[:num_samples, 1]

    # Filter annotations within the first 10 seconds
    annotations_in_range = [(idx, label) for idx, label in zip(sample_idx, labels) if idx < num_samples]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Channel 1
    plt.subplot(2, 1, 1)
    plt.plot(time, channel_1, label="Channel 1", color="blue")
    plt.title("ECG Signal - First 10 Seconds (Channel 1)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Add annotations below the graph for Channel 1
    for idx, label in annotations_in_range:
        plt.text(idx / fs, min(channel_1) - abs(min(channel_1)) * 0.1,
                 label, fontsize=8, color="red", ha="center")
    
    # Channel 2
    plt.subplot(2, 1, 2)
    plt.plot(time, channel_2, label="Channel 2", color="green")
    plt.title("ECG Signal - First 10 Seconds (Channel 2)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Add annotations below the graph for Channel 2
    for idx, label in annotations_in_range:
        plt.text(idx / fs, min(channel_2) - abs(min(channel_2)) * 0.1,
                 label, fontsize=8, color="red", ha="center")
    
    plt.tight_layout()
    plt.show()

# Call the function to plot with labels
plot_first_10_seconds_with_labels(signals, fs, sample_idx, labels)
