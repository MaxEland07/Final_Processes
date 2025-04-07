import numpy as np
import matplotlib.pyplot as plt

# Path to dataset
dataset_path = './MIT-BIH Arrhythmia Database/Compiled-Data/ecg_denoising_train.npz'

# Load metadata first (lightweight)
with np.load(dataset_path, allow_pickle=True) as data:
    n_samples = data['noisy'].shape[0]
    metadata = data['metadata']

# Pick a random sample
sample_index = np.random.randint(0, n_samples)

# Load only the required sample using memory mapping
with np.load(dataset_path, mmap_mode='r') as data:
    noisy_sample = data['noisy'][sample_index]
    clean_sample = data['clean'][sample_index]

# Extract metadata for the selected sample
meta = metadata[sample_index]  # Unwrap the object
record_id = meta.get('record_id', 'Unknown')
noise_type = meta.get('noise_type', 'Unknown')
snr = meta.get('snr', 'Unknown')

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(noisy_sample)
plt.title(f'Noisy ECG Sample | Record: {record_id} | Noise: {noise_type} | SNR: {snr} dB')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(clean_sample)
plt.title('Clean ECG Sample')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()