import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from sklearn.metrics import mean_squared_error

# Define directories
data_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(data_dir, "Processed-Data")
noise_dir = os.path.join(data_dir, "Noise-Data")
denoised_dir = os.path.join(data_dir, "Denoised-Data")
results_dir = "./Results"
vis_dir = "./Visulisations/Filtering_Results"

# Create necessary directories
for directory in [denoised_dir, results_dir, vis_dir]:
    os.makedirs(directory, exist_ok=True)

# Define SNR and MSE calculation functions
def calculate_snr(clean_signal, noisy_signal, denoised_signal=None):
    """Calculate SNR for noisy and denoised signals in dB"""
    signal_power = np.mean(clean_signal**2)
    noise = noisy_signal - clean_signal
    noise_power = np.mean(noise**2)
    snr_noisy = 10 * np.log10(signal_power / noise_power)
    
    if denoised_signal is not None:
        residual_noise = denoised_signal - clean_signal
        residual_noise_power = np.mean(residual_noise**2)
        snr_denoised = 10 * np.log10(signal_power / residual_noise_power)
        return snr_noisy, snr_denoised
    return snr_noisy

def calculate_mse(clean_signal, noisy_signal, denoised_signal=None):
    """Calculate MSE for noisy and denoised signals"""
    mse_noisy = mean_squared_error(clean_signal, noisy_signal)
    if denoised_signal is not None:
        mse_denoised = mean_squared_error(clean_signal, denoised_signal)
        return mse_noisy, mse_denoised
    return mse_noisy

# Define filtering techniques
def apply_butterworth(data, fs, lowcut=0.5, highcut=45, order=4):
    """Apply Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def apply_savitzky_golay(data, window_length=15, polyorder=3):
    """Apply Savitzky-Golay filter"""
    return signal.savgol_filter(data, window_length, polyorder, axis=0)

def apply_wavelet_denoising(data, wavelet='db4', level=4):
    """Apply wavelet denoising"""
    from pywt import wavedec, waverec, threshold
    denoised = np.zeros_like(data)
    for channel in range(data.shape[1]):
        coeffs = wavedec(data[:, channel], wavelet, level=level)
        for i in range(1, len(coeffs)):
            coeffs[i] = threshold(coeffs[i], value=np.std(coeffs[i])/2, mode='soft')
        denoised[:, channel] = waverec(coeffs, wavelet)
    return denoised

def apply_moving_average(data, window_size=5):
    """Apply moving average filter"""
    kernel = np.ones((window_size,)) / window_size
    denoised = np.zeros_like(data)
    for channel in range(data.shape[1]):
        denoised[:, channel] = np.convolve(data[:, channel], kernel, mode='same')
    return denoised

# Filter techniques dictionary
filter_techniques = {
    "Butterworth": apply_butterworth,
    "Savitzky-Golay": apply_savitzky_golay,
    "Wavelet": apply_wavelet_denoising,
    "Moving Average": apply_moving_average
}

# Main function to process Record 100 at all SNRs
def process_record(record_id="100", snr_levels=[0, 6, 12, -6]):
    """Process Record 100 across all noise types and specified SNR levels"""
    results = []
    
    # Load clean signal
    clean_record_path = os.path.join(processed_dir, f"{record_id}_full.npz")
    if not os.path.exists(clean_record_path):
        print(f"Clean record {record_id}_full.npz not found in {processed_dir}. Please run Process-Data.py first.")
        return None
    
    clean_data = np.load(clean_record_path)
    clean_signal = clean_data["signals"]
    fs = clean_data["fs"]
    
    # Results dataframe
    columns = ["Record_ID", "Noise_Type", "SNR_Level", "Filter_Technique", 
               "Original_SNR", "Improved_SNR", "SNR_Gain", "Original_MSE", "Improved_MSE"]
    results_df = pd.DataFrame(columns=columns)
    
    # Process each noise type
    for noise_type in os.listdir(noise_dir):
        noise_config_dir = os.path.join(noise_dir, noise_type)
        if not os.path.isdir(noise_config_dir):
            continue
        
        # Process each SNR level
        for snr in snr_levels:
            noisy_file = f"record_{record_id}_full_{snr}dB.npz"
            noisy_path = os.path.join(noise_config_dir, noisy_file)
            
            if not os.path.exists(noisy_path):
                print(f"Noisy file {noisy_file} not found for {noise_type} at {snr}dB. Skipping.")
                continue
            
            noisy_data = np.load(noisy_path)
            noisy_signal = noisy_data["signals"]
            
            # Apply each filter
            for filter_name, filter_func in filter_techniques.items():
                print(f"Processing Record {record_id}: {noise_type} at {snr}dB with {filter_name}")
                
                # Apply filter
                if filter_name == "Butterworth":
                    denoised_signal = filter_func(noisy_signal, fs)
                else:
                    denoised_signal = filter_func(noisy_signal)
                
                # Calculate metrics
                orig_snr, imp_snr = calculate_snr(clean_signal, noisy_signal, denoised_signal)
                snr_gain = imp_snr - orig_snr
                orig_mse, imp_mse = calculate_mse(clean_signal, noisy_signal, denoised_signal)
                
                # Store results
                results_df = pd.concat([results_df, pd.DataFrame([{
                    "Record_ID": record_id,
                    "Noise_Type": noise_type,
                    "SNR_Level": snr,
                    "Filter_Technique": filter_name,
                    "Original_SNR": orig_snr,
                    "Improved_SNR": imp_snr,
                    "SNR_Gain": snr_gain,
                    "Original_MSE": orig_mse,
                    "Improved_MSE": imp_mse
                }])], ignore_index=True)
                
                # Save denoised signal and visualization
                output_dir = os.path.join(denoised_dir, noise_type, filter_name)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{record_id}_{snr}dB_{filter_name}.npz")
                
                np.savez_compressed(
                    output_path,
                    signals=denoised_signal,
                    sample_indices=noisy_data["sample_indices"],
                    labels=noisy_data["labels"],
                    fs=fs
                )
                
                visualize_results(clean_signal, noisy_signal, denoised_signal, 
                                noise_type, filter_name, snr, record_id, 
                                orig_snr, imp_snr, orig_mse, imp_mse)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(results_dir, f"filter_performance_{record_id}.csv"), index=False)
    
    # Display table in console
    print("\nFiltering Results Table:")
    print(results_df.to_string(index=False))
    
    plot_summary_results(results_df, record_id)
    
    return results_df

def visualize_results(clean, noisy, denoised, noise_type, filter_name, snr_level, 
                     record_id, orig_snr, imp_snr, orig_mse, imp_mse):
    """Generate visualization of filtering results"""
    segment_start = 1000
    segment_length = 2000
    segment_end = segment_start + segment_length
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(clean[segment_start:segment_end, 0], label='Clean')
    plt.title(f'Record {record_id} - Clean ECG Signal')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(noisy[segment_start:segment_end, 0], label=f'Noisy (SNR: {orig_snr:.2f} dB, MSE: {orig_mse:.4f})')
    plt.title(f'Noisy ECG - {noise_type} at {snr_level} dB')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(denoised[segment_start:segment_end, 0], 
             label=f'Denoised (SNR: {imp_snr:.2f} dB, MSE: {imp_mse:.4f})')
    plt.title(f'Denoised ECG using {filter_name} (SNR Gain: {imp_snr-orig_snr:.2f} dB)')
    plt.ylabel('Amplitude')
    plt.xlabel('Samples')
    plt.legend()
    
    plt.tight_layout()
    
    output_dir = os.path.join(vis_dir, noise_type)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{record_id}_{filter_name}_{snr_level}dB.png'), dpi=300)
    plt.close()

def plot_summary_results(results_df, record_id):
    """Generate summary plots"""
    # SNR Gain by Filter and SNR Level
    avg_by_filter_snr = results_df.groupby(['Filter_Technique', 'SNR_Level'])['SNR_Gain'].mean().reset_index()
    pivot_df = avg_by_filter_snr.pivot(index='SNR_Level', columns='Filter_Technique', values='SNR_Gain')
    
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title(f'Record {record_id} - SNR Improvement by Filter and SNR Level')
    plt.xlabel('Original SNR Level (dB)')
    plt.ylabel('SNR Improvement (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'filter_performance_snr_{record_id}.png'), dpi=300)
    plt.close()
    
    # SNR Gain by Filter and Noise Type
    avg_by_filter_noise = results_df.groupby(['Filter_Technique', 'Noise_Type'])['SNR_Gain'].mean().reset_index()
    pivot_df2 = avg_by_filter_noise.pivot(index='Noise_Type', columns='Filter_Technique', values='SNR_Gain')
    
    plt.figure(figsize=(14, 10))
    pivot_df2.plot(kind='bar', ax=plt.gca())
    plt.title(f'Record {record_id} - SNR Improvement by Filter and Noise Type')
    plt.xlabel('Noise Type')
    plt.ylabel('SNR Improvement (dB)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'filter_performance_noise_{record_id}.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    record_id = "100"  # Hardcoded to Record 100
    snr_levels = [0, 6, 12, -6]  # Specify all SNR levels to process
    print(f"Starting ECG signal denoising for Record {record_id} at SNR levels: {snr_levels} dB...")
    results = process_record(record_id, snr_levels)
    if results is not None:
        print(f"Denoising complete. Results saved to filter_performance_{record_id}.csv")
    else:
        print("Denoising failed due to missing data.")
