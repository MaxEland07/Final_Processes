import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

# === CONFIGURATION ===
model_path = "./Results/lstm_denoiser_all_noise_types_best.keras"
val_file = "./MIT-BIH Arrhythmia Database/Compiled-Data/100_all_noise_types_valid.npz"
n_samples_to_plot = 5  # How many ECG windows to test

# === LOAD MODEL ===
print("Loading model...")
model = load_model(model_path)

# === LOAD VALIDATION DATA ===
print("Loading validation data...")
data = np.load(val_file)
X_val = data["noisy"]
y_val = data["clean"]

# Reshape if needed (to [batch, 512, 1])
if X_val.ndim == 2:
    X_val = X_val[..., np.newaxis]
    y_val = y_val[..., np.newaxis]

print(f"Validation samples loaded: {X_val.shape[0]}")

# === SNR FUNCTION ===
def calculate_snr(signal, noise):
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    SNR = 10 * log10(P_signal / P_noise)
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')  # Perfect reconstruction
    return 10 * np.log10(signal_power / noise_power)

# === PREDICT AND DISPLAY ===
print("Displaying predictions and SNR improvements...\n")
sample_indices = np.random.choice(len(X_val), n_samples_to_plot, replace=False)

for i, idx in enumerate(sample_indices):
    noisy_input = X_val[idx]
    clean_target = y_val[idx]

    # Predict
    prediction = model.predict(np.expand_dims(noisy_input, axis=0), verbose=0)[0]

    # Compute SNRs
    snr_before = calculate_snr(clean_target, clean_target - noisy_input)
    snr_after = calculate_snr(clean_target, clean_target - prediction)
    improvement = snr_after - snr_before

    # Print SNR info
    print(f"Sample {idx}:")
    print(f"  SNR before: {snr_before:.2f} dB")
    print(f"  SNR after:  {snr_after:.2f} dB")
    print(f"  Improvement: {improvement:.2f} dB\n")

    # Plot the results
    plt.figure(figsize=(12, 4))
    plt.plot(noisy_input.squeeze(), label='Noisy Input', alpha=0.6)
    plt.plot(clean_target.squeeze(), label='Clean Target', alpha=0.6)
    plt.plot(prediction.squeeze(), label='Denoised Output', alpha=0.9)
    plt.title(f"ECG Denoising - Sample {idx}\nSNR Before: {snr_before:.2f} dB | After: {snr_after:.2f} dB")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
