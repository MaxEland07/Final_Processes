import os
import numpy as np
# import wfdb # Not used in this specific snippet, but likely needed elsewhere

# --- Configuration ---
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data") # Path to clean data
noisy_dir = os.path.join(base_dir, "Noise-Data") # Path to noisy data (adjust if needed)
windowed_dir = os.path.join(base_dir, "Windowed-Data") # Output directory

record_id = "100" # Example record ID
noise_type = "MA" # Example noise type (make sure this subfolder exists in noisy_dir)
snr = 0 # Example SNR level

noisy_file_name = f"{record_id}_full_{noise_type}_{snr}dB.npz" # Example noisy filename (adjust format if needed)
clean_file_name = f"{record_id}_full.npz" # Corresponding clean filename

noisy_file_path = os.path.join(noisy_dir, noise_type, noisy_file_name)
clean_file_path = os.path.join(processed_dir, clean_file_name)
output_file_path = os.path.join(windowed_dir, f"{record_id}_{noise_type}_{snr}dB_win512_stride256.npz")

window_size = 512
stride = 256

# --- Create Output Directory ---
if not os.path.exists(windowed_dir):
    os.makedirs(windowed_dir, exist_ok=True)
    print(f"Created directory: {windowed_dir}")

# --- Check if files exist ---
if not os.path.exists(noisy_file_path):
    print(f"ERROR: Noisy file not found at {noisy_file_path}")
    exit()
if not os.path.exists(clean_file_path):
    print(f"ERROR: Clean file not found at {clean_file_path}")
    exit()

# --- Load Data ---
print(f"Loading noisy data: {noisy_file_path}")
noisy_data = np.load(noisy_file_path)
noisy_signals_full = noisy_data['signals'] # Shape: (total_samples, num_channels)
fs = noisy_data['fs']

print(f"Loading clean data: {clean_file_path}")
clean_data = np.load(clean_file_path)
clean_signals_full = clean_data['signals'] # Shape: (total_samples, num_channels)

# Ensure signals have the same length (they should if generated correctly)
assert noisy_signals_full.shape[0] == clean_signals_full.shape[0], \
    f"Signal lengths differ: Noisy={noisy_signals_full.shape[0]}, Clean={clean_signals_full.shape[0]}"
assert noisy_signals_full.shape[1] == clean_signals_full.shape[1], \
    f"Number of channels differ: Noisy={noisy_signals_full.shape[1]}, Clean={clean_signals_full.shape[1]}"

signal_length = noisy_signals_full.shape[0]
num_channels = noisy_signals_full.shape[1]

# --- Windowing ---
noisy_windows_list = []
clean_windows_list = []

print(f"Creating overlapping windows (size={window_size}, stride={stride})...")
# Process each channel separately first
for channel in range(num_channels):
    print(f"  Processing Channel {channel+1}/{num_channels}")
    noisy_signal_ch = noisy_signals_full[:, channel]
    clean_signal_ch = clean_signals_full[:, channel]

    # Slide the window across the signal for this channel
    start = 0
    while start + window_size <= signal_length:
        end = start + window_size

        # Extract the window for both noisy and clean signals
        noisy_window = noisy_signal_ch[start:end]
        clean_window = clean_signal_ch[start:end]

        # Reshape window to (window_size, 1) as expected by LSTM
        noisy_window_reshaped = noisy_window.reshape(window_size, 1)
        clean_window_reshaped = clean_window.reshape(window_size, 1)

        # Append the reshaped windows to the lists
        noisy_windows_list.append(noisy_window_reshaped)
        clean_windows_list.append(clean_window_reshaped)

        # Move window start by the stride
        start += stride

# Convert lists of windows to numpy arrays
# The shape will be (total_num_windows_across_all_channels, window_size, 1)
noisy_windows_array = np.array(noisy_windows_list)
clean_windows_array = np.array(clean_windows_list)

print(f"Generated {noisy_windows_array.shape[0]} windows.")
print(f"Shape of noisy windows array: {noisy_windows_array.shape}") # Should be (N, 512, 1)
print(f"Shape of clean windows array: {clean_windows_array.shape}") # Should be (N, 512, 1)

# --- Save the windowed signals ---
print(f"Saving windowed data to: {output_file_path}")
np.savez(
    output_file_path,
    noisy_windows=noisy_windows_array, # Key for noisy data
    clean_windows=clean_windows_array, # Key for clean data (target)
    fs=fs # Save sampling frequency if needed later
)

print("Windowing complete.")