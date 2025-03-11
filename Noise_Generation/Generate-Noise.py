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

# Use record 100
records = ["100_full.npz"]

# Define SNR levels (in dB) - from very noisy to clean
snr_levels = [-6, 0, 6, 12, 18, 24]

# Define noise types with abbreviations
noise_types = {
    "Gaussian": "G",
    "Baseline Wander": "BW",
    "Power Line": "PL",
    "Muscle Artifact": "MA",
    "Electrode Motion": "EM"
}

# Define configurations (single noise types and combinations)
configs = [
    ["Gaussian"],
    ["Baseline Wander"],
    ["Power Line"],
    ["Muscle Artifact"],
    ["Electrode Motion"],
    ["Baseline Wander", "Muscle Artifact"],
    ["Power Line", "Gaussian"],
    ["Baseline Wander", "Power Line"],
    ["Gaussian", "Baseline Wander", "Power Line", "Muscle Artifact", "Electrode Motion"]
]

# Noise generation functions with power normalization

def generate_baseline_wander(signal_shape, p_noise, fs):
    t = np.arange(signal_shape[0]) / fs
    freqs = [0.05, 0.1, 0.2, 0.3]
    power_distribution = [0.3, 0.3, 0.25, 0.15]
    
    noise = np.zeros(signal_shape[0])
    for i, freq in enumerate(freqs):
        component_power = p_noise * power_distribution[i]
        amplitude = np.sqrt(2 * component_power)
        noise += amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
    
    noise = noise[:, np.newaxis]
    noise = np.repeat(noise, signal_shape[1], axis=1)
    
    # Normalize to match target power
    p_noise_actual = np.var(noise[:, 0])
    noise *= np.sqrt(p_noise / p_noise_actual) if p_noise_actual > 0 else 1
    return noise

def generate_power_line_interference(signal_shape, p_noise, fs):
    t = np.arange(signal_shape[0]) / fs
    f_line = 50
    harmonics = [1, 2, 3]
    power_distribution = [0.7, 0.2, 0.1]
    
    noise = np.zeros(signal_shape[0])
    for i, h in enumerate(harmonics):
        component_power = p_noise * power_distribution[i]
        amplitude = np.sqrt(2 * component_power)
        freq_deviation = np.random.uniform(-0.1, 0.1)
        am_freq = np.random.uniform(0.1, 1.0)
        am_depth = np.random.uniform(0.05, 0.15)
        am_component = 1 + am_depth * np.sin(2 * np.pi * am_freq * t)
        noise += amplitude * am_component * np.sin(2 * np.pi * (f_line * h + freq_deviation) * t)
    
    noise = noise[:, np.newaxis]
    noise = np.repeat(noise, signal_shape[1], axis=1)
    
    # Normalize to match target power
    p_noise_actual = np.var(noise[:, 0])
    noise *= np.sqrt(p_noise / p_noise_actual) if p_noise_actual > 0 else 1
    return noise

def generate_muscle_artifact(signal_shape, p_noise, fs):
    white_noise = np.random.normal(0, 1, signal_shape[0])
    fft_noise = np.fft.rfft(white_noise)
    freqs = np.fft.rfftfreq(signal_shape[0], 1/fs)
    
    filter_shape = np.ones_like(freqs)
    filter_shape[freqs < 20] = np.linspace(0.1, 1, np.sum(freqs < 20))
    filter_shape[freqs > 100] = np.exp(-(freqs[freqs > 100] - 100) / 100)
    
    filtered_fft = fft_noise * filter_shape
    filtered_noise = np.fft.irfft(filtered_fft)
    filtered_noise = filtered_noise / np.std(filtered_noise) * np.sqrt(p_noise)
    
    t = np.arange(signal_shape[0]) / fs
    num_bursts = np.random.randint(3, 8)
    envelope = np.zeros_like(t)
    
    for _ in range(num_bursts):
        center = np.random.uniform(0, t[-1])
        width = np.random.uniform(0.5, 2.0)
        height = np.random.uniform(0.7, 1.0)
        envelope += height * np.exp(-((t - center) / width) ** 2)
    
    envelope = np.clip(envelope, 0.2, 1.0)
    filtered_noise *= envelope
    
    noise = filtered_noise[:, np.newaxis]
    noise = np.repeat(noise, signal_shape[1], axis=1)
    
    # Normalize to match target power
    p_noise_actual = np.var(noise[:, 0])
    noise *= np.sqrt(p_noise / p_noise_actual) if p_noise_actual > 0 else 1
    return noise

def generate_electrode_motion_artifact(signal_shape, p_noise, fs):
    t = np.arange(signal_shape[0]) / fs
    noise = np.zeros(signal_shape[0])
    
    num_movements = np.random.randint(3, 10)
    for _ in range(num_movements):
        movement_time = np.random.randint(0, signal_shape[0])
        amplitude = np.random.uniform(0.5, 2.0) * np.sqrt(p_noise)
        decay_rate = np.random.uniform(0.1, 1.0)
        step = np.zeros(signal_shape[0])
        step[movement_time:] = 1
        decay = np.exp(-decay_rate * (t - t[movement_time]) * (t >= t[movement_time]))
        noise += amplitude * step * decay
    
    high_freq_noise = np.random.normal(0, np.sqrt(p_noise) * 0.3, signal_shape[0])
    envelope = np.clip(np.abs(np.gradient(noise)) * 10, 0, 1)
    high_freq_component = high_freq_noise * envelope
    
    noise += high_freq_component
    noise = noise[:, np.newaxis]
    noise = np.repeat(noise, signal_shape[1], axis=1)
    
    # Normalize to match target power
    p_noise_actual = np.var(noise[:, 0])
    noise *= np.sqrt(p_noise / p_noise_actual) if p_noise_actual > 0 else 1
    return noise

def generate_combined_noise(noise_types, signal_shape, p_noise_total, fs):
    power_distribution = {
        "Gaussian": 0.1,
        "Baseline Wander": 0.3,
        "Power Line": 0.2,
        "Muscle Artifact": 0.25,
        "Electrode Motion": 0.15
    }
    
    selected_distribution = {k: power_distribution[k] for k in noise_types}
    total = sum(selected_distribution.values())
    normalized_distribution = {k: v/total for k, v in selected_distribution.items()}
    
    total_noise = np.zeros(signal_shape)
    for noise_type, power_fraction in normalized_distribution.items():
        p_noise_individual = p_noise_total * power_fraction
        
        if noise_type == "Gaussian":
            noise = np.random.normal(0, np.sqrt(p_noise_individual), signal_shape)
        elif noise_type == "Baseline Wander":
            noise = generate_baseline_wander(signal_shape, p_noise_individual, fs)
        elif noise_type == "Power Line":
            noise = generate_power_line_interference(signal_shape, p_noise_individual, fs)
        elif noise_type == "Muscle Artifact":
            noise = generate_muscle_artifact(signal_shape, p_noise_individual, fs)
        elif noise_type == "Electrode Motion":
            noise = generate_electrode_motion_artifact(signal_shape, p_noise_individual, fs)
        
        total_noise += noise
    
    # Normalize combined noise to match total power
    p_noise_actual = np.var(total_noise[:, 0])
    total_noise *= np.sqrt(p_noise_total / p_noise_actual) if p_noise_actual > 0 else 1
    return total_noise

# Main loop to generate noisy data
for record in records:
    # Load clean signal
    record_path = os.path.join(processed_dir, record)
    data = np.load(record_path)
    clean_signal = data["signals"]  # Shape: [n_samples, 2]
    fs = data["fs"]                 # Sampling frequency (e.g., 360 Hz)

    # Compute signal power using the average variance across both channels
    p_signal = np.mean(np.var(clean_signal, axis=0))

    for config in configs:
        # Generate config name (e.g., "G", "BW_MA")
        config_name = "_".join([noise_types[nt] for nt in config])
        config_dir = os.path.join(noise_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        for snr_level in snr_levels:
            # Compute total noise power for the desired SNR
            p_noise_total = p_signal / (10 ** (snr_level / 10))

            # Process each channel separately
            noisy_signals = np.zeros_like(clean_signal)
            for ch in range(clean_signal.shape[1]):
                # Generate noise for the current channel with independent random seed
                np.random.seed(np.random.randint(1000) + ch)
                total_noise = generate_combined_noise(config, (clean_signal.shape[0], 1), p_noise_total, fs)
                noisy_signals[:, ch] = clean_signal[:, ch] + total_noise[:, 0]

            # Verify and adjust noise power to ensure target SNR
            noise_component = noisy_signals - clean_signal
            p_noise_actual = np.mean(np.var(noise_component, axis=0))
            if p_noise_actual > 0:
                scaling_factor = np.sqrt(p_noise_total / p_noise_actual)
                noisy_signals = clean_signal + noise_component * scaling_factor

            # Save to file
            record_name = record.replace(".npz", "")
            output_file = f"record_{record_name}_{snr_level}dB.npz"
            output_path = os.path.join(config_dir, output_file)
            np.savez_compressed(
                output_path,
                signals=noisy_signals,
                sample_indices=data["sample_indices"],
                labels=data["labels"],
                fs=fs
            )
            # Verify actual SNR
            p_noise_actual = np.mean(np.var(noisy_signals - clean_signal, axis=0))
            actual_snr = 10 * np.log10(p_signal / p_noise_actual)
            print(f"Saved {output_file} in {config_dir}, Target SNR: {snr_level} dB, Actual SNR: {actual_snr:.2f} dB")

print("Noise generation complete.")