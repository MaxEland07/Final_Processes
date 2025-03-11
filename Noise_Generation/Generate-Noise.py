import numpy as np
import os
import shutil

# Define Directories
data_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(data_dir, "Processed-Data")
noise_dir = os.path.join(data_dir, "Noise-Data")

# Clear and recreate noise_dir
if os.path.exists(noise_dir):
    shutil.rmtree(noise_dir)
os.makedirs(noise_dir, exist_ok=True)

# Use first 3 records for testing (adjust as needed)
records = os.listdir(processed_dir)[:1]

# Define SNR levels (in dB)
snr_levels = [-6, 0, 6, 12, 18, 24]

# Define noise types with abbreviations
noise_types = {
    "Gaussian": "G",
    "Baseline Wander": "BW",
    "Power Line": "PL",
    "Muscle Artifact": "MA"
}

# Define configurations (single noise types and combinations)
configs = [
    ["Gaussian"],
    ["Baseline Wander"],
    ["Power Line"],
    ["Muscle Artifact"],
    ["Baseline Wander", "Muscle Artifact"],
    ["Power Line", "Gaussian"],
    ["Baseline Wander", "Power Line"]
]

# Noise generation function
def generate_noise(noise_type, signal_shape, p_noise, fs):
    """
    Generate noise of a specific type with given power.
    
    Args:
        noise_type (str): Type of noise (e.g., 'Gaussian').
        signal_shape (tuple): Shape of the signal [n_samples, n_channels].
        p_noise (float): Desired noise power.
        fs (float): Sampling frequency (Hz).
    
    Returns:
        np.ndarray: Generated noise array.
    """
    if noise_type == "Gaussian":
        noise = np.random.normal(0, np.sqrt(p_noise), signal_shape)

    elif noise_type == "Baseline Wander":
        t = np.arange(signal_shape[0]) / fs
        freq = np.random.uniform(0.1, 0.5)  # Low frequency between 0.1-0.5 Hz
        amplitude = np.sqrt(2 * p_noise)    # Power = A^2 / 2
        noise = amplitude * np.sin(2 * np.pi * freq * t)[:, np.newaxis]
        noise = np.repeat(noise, signal_shape[1], axis=1)

    elif noise_type == "Power Line":
        t = np.arange(signal_shape[0]) / fs
        freq = 50  # 50 Hz (use 50 Hz for regions with 50 Hz power)
        amplitude = np.sqrt(2 * p_noise)
        noise = amplitude * np.sin(2 * np.pi * freq * t)[:, np.newaxis]
        noise = np.repeat(noise, signal_shape[1], axis=1)

    elif noise_type == "Muscle Artifact":
        noise = np.random.normal(0, np.sqrt(p_noise), signal_shape)
        noise = np.diff(noise, axis=0, prepend=noise[0])  # High-pass effect

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noise

# Main loop to generate noisy data
for record in records:
    # Load clean signal
    record_path = os.path.join(processed_dir, record)
    data = np.load(record_path)
    clean_signal = data["signals"]  # Shape: [n_samples, 2]
    fs = data["fs"]                 # Sampling frequency (e.g., 360 Hz)

    # Compute signal power (average variance across channels)
    p_signal = np.mean(np.var(clean_signal, axis=0))

    for config in configs:
        # Generate config name (e.g., "G", "BW_MA")
        config_name = "_".join([noise_types[nt] for nt in config])
        config_dir = os.path.join(noise_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        for snr in snr_levels:
            # Compute total noise power for the desired SNR
            p_noise_total = p_signal / (10 ** (snr / 10))
            # Divide noise power equally among components
            p_noise_individual = p_noise_total / len(config)

            # Generate and sum noise for each type in the config
            total_noise = np.zeros_like(clean_signal)
            for noise_type in config:
                noise = generate_noise(noise_type, clean_signal.shape, p_noise_individual, fs)
                total_noise += noise

            # Create noisy signal
            noisy_signal = clean_signal + total_noise

            # Save to file
            record_name = record.replace(".npz", "")
            output_file = f"record_{record_name}_{snr}dB.npz"
            output_path = os.path.join(config_dir, output_file)
            np.savez_compressed(
                output_path,
                signals=noisy_signal,
                sample_indices=data["sample_indices"],
                labels=data["labels"],
                fs=fs
            )
            print(f"Saved {output_file} in {config_dir}")

print("Noise generation complete!")