import os
import numpy as np
import wfdb

# Define directories
base_dir = "/kaggle/working/MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data")
noisy_dir = os.path.join(processed_dir, "noisy_data")
stress_test_dir = os.path.join(base_dir, "stress_test")

# Create noisy_data directory if it doesn't exist
os.makedirs(noisy_dir, exist_ok=True)

# Function to add noise and compute SNR
def add_noise(clean_signal, noise, target_snr):
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(noise**2)
    scale_factor = np.sqrt(signal_power / (noise_power * 10**(target_snr / 10)))
    noisy_signal = clean_signal + noise * scale_factor
    return noisy_signal

# Load processed data
input_file = os.path.join(processed_dir, "100_full.npz")
data = np.load(input_file)
filtered_signals = data['signals']  # Shape: (samples, channels)
fs = data['fs']  # Sampling frequency
sample_indices = data['sample_indices']
labels = data['labels']

# Read noise signals from stress_test directory
noise_files = ['bw', 'ma', 'em']
noises = {}
for noise_type in noise_files:
    noise_sig, fields = wfdb.rdsamp(os.path.join(stress_test_dir, noise_type))
    noises[noise_type] = noise_sig[:, 0]  # Use first channel, match length if needed
    if len(noises[noise_type]) < len(filtered_signals):
        noises[noise_type] = np.pad(noises[noise_type], (0, len(filtered_signals) - len(noises[noise_type])), 'wrap')

# Add noise at SNR = 6 dB
snr_value = 6  # dB
for noise_type in noise_files:
    noisy_signals = np.zeros_like(filtered_signals)
    noise = noises[noise_type][:len(filtered_signals)]  # Ensure same length
    for channel in range(filtered_signals.shape[1]):
        noisy_signals[:, channel] = add_noise(filtered_signals[:, channel], noise, snr_value)
    
    # Save noisy data
    output_file = os.path.join(noisy_dir, f"100_full_{noise_type}_snr{snr_value}.npz")
    np.savez(output_file,
             signals=noisy_signals,
             sample_indices=sample_indices,
             labels=labels,
             fs=fs)
    print(f"Saved noisy data to {output_file} (shape: {noisy_signals.shape})")

print(f"Noise addition complete! Noisy files saved in {noisy_dir}")