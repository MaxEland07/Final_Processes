import os
import numpy as np
# import wfdb # Not used in this specific snippet, but likely needed elsewhere

# --- Configuration ---
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data") # Path to clean data
noisy_dir = os.path.join(base_dir, "Noisy-Data") # Path to noisy data (adjust if needed)
windowed_dir = os.path.join(base_dir, "Windowed-Data") # Output directory
compiled_dir = os.path.join(base_dir, "Compiled-Data")  # New directory for compiled datasets

# --- Create Output Directories ---
for directory in [windowed_dir, compiled_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Configuration parameters
record_id = "100" # Example record ID
noise_types = ["ma", "bw", "em"] # Example noise type (make sure this subfolder exists in noisy_dir)
snrs = [-6, 0, 6] # Example SNR levels
window_size = 512
stride = 256

def compile_datasets(train_split=0.8, valid_split=0.1, test_split=0.1, separate_noise_types=False):
    """
    Compile all windowed data into train/valid/test datasets for easier model training.
    
    Args:
        train_split: Fraction of data to use for training (default 0.8)
        valid_split: Fraction of data to use for validation (default 0.1)
        test_split: Fraction of data to use for testing (default 0.1)
        separate_noise_types: If True, create separate datasets for each noise type
    """
    assert abs(train_split + valid_split + test_split - 1.0) < 1e-9, "Splits must sum to 1.0"
    
    if separate_noise_types:
        # Create separate datasets for each noise type
        for noise_type in noise_types:
            print(f"\nCompiling datasets for {noise_type} noise...")
            all_noisy_windows = []
            all_clean_windows = []
            all_snr_values = []  # Track SNR values
            
            for snr in snrs:
                window_file = os.path.join(windowed_dir, f"{record_id}_{noise_type}_{snr}dB_win512_stride256.npz")
                if os.path.exists(window_file):
                    data = np.load(window_file)
                    n_windows = data["noisy_windows"].shape[0]
                    all_noisy_windows.append(data["noisy_windows"])
                    all_clean_windows.append(data["clean_windows"])
                    all_snr_values.extend([snr] * n_windows)  # Add SNR value for each window
                    print(f"  Added {n_windows} windows from SNR {snr}dB")
            
            if not all_noisy_windows:
                print(f"  No data found for {noise_type} noise type, skipping")
                continue
                
            # Concatenate all windows
            all_noisy = np.concatenate(all_noisy_windows, axis=0)
            all_clean = np.concatenate(all_clean_windows, axis=0)
            all_snr_values = np.array(all_snr_values)
            
            # Shuffle the data (with same seed for all arrays)
            indices = np.arange(len(all_noisy))
            np.random.seed(42)
            np.random.shuffle(indices)
            all_noisy = all_noisy[indices]
            all_clean = all_clean[indices]
            all_snr_values = all_snr_values[indices]
            
            # Split into train/valid/test
            n_samples = len(all_noisy)
            train_end = int(n_samples * train_split)
            valid_end = train_end + int(n_samples * valid_split)
            
            train_noisy = all_noisy[:train_end]
            train_clean = all_clean[:train_end]
            train_snrs = all_snr_values[:train_end]
            
            valid_noisy = all_noisy[train_end:valid_end]
            valid_clean = all_clean[train_end:valid_end]
            valid_snrs = all_snr_values[train_end:valid_end]
            
            test_noisy = all_noisy[valid_end:]
            test_clean = all_clean[valid_end:]
            test_snrs = all_snr_values[valid_end:]
            
            # Save the compiled datasets
            output_base = os.path.join(compiled_dir, f"{record_id}_{noise_type}")
            
            np.savez(f"{output_base}_train.npz", 
                     noisy=train_noisy, clean=train_clean, snr=train_snrs)
            np.savez(f"{output_base}_valid.npz", 
                     noisy=valid_noisy, clean=valid_clean, snr=valid_snrs)
            np.savez(f"{output_base}_test.npz", 
                     noisy=test_noisy, clean=test_clean, snr=test_snrs)
            
            print(f"  Saved compiled datasets for {noise_type}:")
            print(f"    Train: {train_noisy.shape[0]} samples")
            print(f"    Valid: {valid_noisy.shape[0]} samples") 
            print(f"    Test:  {test_noisy.shape[0]} samples")
    
    else:
        # Create combined dataset for all noise types
        print("\nCompiling combined dataset for all noise types...")
        all_noisy_windows = []
        all_clean_windows = []
        noise_type_indices = []  # To track which noise type each window belongs to
        all_snr_values = []      # To track SNR values for each window
        
        for i, noise_type in enumerate(noise_types):
            for snr in snrs:
                window_file = os.path.join(windowed_dir, f"{record_id}_{noise_type}_{snr}dB_win512_stride256.npz")
                if os.path.exists(window_file):
                    data = np.load(window_file)
                    n_windows = data["noisy_windows"].shape[0]
                    all_noisy_windows.append(data["noisy_windows"])
                    all_clean_windows.append(data["clean_windows"])
                    noise_type_indices.extend([i] * n_windows)  # Add noise type index for each window
                    all_snr_values.extend([snr] * n_windows)    # Add SNR value for each window
                    print(f"  Added {n_windows} windows from {noise_type} SNR {snr}dB")
        
        # Concatenate all windows
        all_noisy = np.concatenate(all_noisy_windows, axis=0)
        all_clean = np.concatenate(all_clean_windows, axis=0)
        noise_type_indices = np.array(noise_type_indices)
        all_snr_values = np.array(all_snr_values)
        
        # Create a metadata dictionary mapping indices to noise type names
        noise_type_map = {i: noise_type for i, noise_type in enumerate(noise_types)}
        
        # Shuffle the data (with same seed for all arrays)
        indices = np.arange(len(all_noisy))
        np.random.seed(42)
        np.random.shuffle(indices)
        all_noisy = all_noisy[indices]
        all_clean = all_clean[indices]
        noise_type_indices = noise_type_indices[indices]
        all_snr_values = all_snr_values[indices]
        
        # Split into train/valid/test
        n_samples = len(all_noisy)
        train_end = int(n_samples * train_split)
        valid_end = train_end + int(n_samples * valid_split)
        
        train_noisy = all_noisy[:train_end]
        train_clean = all_clean[:train_end]
        train_noise_types = noise_type_indices[:train_end]
        train_snrs = all_snr_values[:train_end]
        
        valid_noisy = all_noisy[train_end:valid_end]
        valid_clean = all_clean[train_end:valid_end]
        valid_noise_types = noise_type_indices[train_end:valid_end]
        valid_snrs = all_snr_values[train_end:valid_end]
        
        test_noisy = all_noisy[valid_end:]
        test_clean = all_clean[valid_end:]
        test_noise_types = noise_type_indices[valid_end:]
        test_snrs = all_snr_values[valid_end:]
        
        # Save the compiled datasets
        output_base = os.path.join(compiled_dir, f"{record_id}_all_noise_types")
        
        np.savez(f"{output_base}_train.npz", 
                 noisy=train_noisy, clean=train_clean, 
                 noise_types=train_noise_types, snrs=train_snrs,
                 noise_type_map=noise_type_map)
                 
        np.savez(f"{output_base}_valid.npz", 
                 noisy=valid_noisy, clean=valid_clean, 
                 noise_types=valid_noise_types, snrs=valid_snrs,
                 noise_type_map=noise_type_map)
                 
        np.savez(f"{output_base}_test.npz", 
                 noisy=test_noisy, clean=test_clean, 
                 noise_types=test_noise_types, snrs=test_snrs,
                 noise_type_map=noise_type_map)
        
        print(f"\nSaved combined datasets:")
        print(f"  Train: {train_noisy.shape[0]} samples")
        print(f"  Valid: {valid_noisy.shape[0]} samples")
        print(f"  Test:  {test_noisy.shape[0]} samples")
        
        # Print distribution statistics
        for split_name, noise_types_array, snrs_array in [
            ("Train", train_noise_types, train_snrs),
            ("Valid", valid_noise_types, valid_snrs),
            ("Test", test_noise_types, test_snrs)
        ]:
            print(f"\n  {split_name} set distribution:")
            for i, noise_type in enumerate(noise_types):
                count = np.sum(noise_types_array == i)
                percentage = 100 * count / len(noise_types_array)
                print(f"    - {noise_type}: {count} samples ({percentage:.1f}%)")
            
            for snr in snrs:
                count = np.sum(snrs_array == snr)
                percentage = 100 * count / len(snrs_array)
                print(f"    - SNR {snr}dB: {count} samples ({percentage:.1f}%)")

# --- Process each noise type and SNR ---
for noise_type in noise_types:
    for snr in snrs:
        noisy_file_name = f"{record_id}_full_{noise_type}_snr{snr}.npz"
        clean_file_name = f"{record_id}_full.npz"

        noisy_file_path = os.path.join(noisy_dir, noisy_file_name)
        clean_file_path = os.path.join(processed_dir, clean_file_name)
        output_file_path = os.path.join(windowed_dir, f"{record_id}_{noise_type}_{snr}dB_win512_stride256.npz")
        
        # --- Check if files exist ---
        if not os.path.exists(noisy_file_path) or not os.path.exists(clean_file_path):
            print(f"ERROR: Files not found for {noise_type} SNR {snr}, skipping...")
            continue

        # --- Load Data ---
        try:
            noisy_data = np.load(noisy_file_path)
            noisy_signals_full = noisy_data['signals']
            fs = noisy_data['fs']

            clean_data = np.load(clean_file_path)
            clean_signals_full = clean_data['signals']

            # Ensure signals have the same shape
            assert noisy_signals_full.shape == clean_signals_full.shape, "Signal shapes differ"
            
            signal_length = noisy_signals_full.shape[0]
            num_channels = noisy_signals_full.shape[1]

            # --- Windowing ---
            noisy_windows_list = []
            clean_windows_list = []

            # Process each channel
            for channel in range(num_channels):
                noisy_signal_ch = noisy_signals_full[:, channel]
                clean_signal_ch = clean_signals_full[:, channel]
                
                # Slide window across the signal
                start = 0
                while start + window_size <= signal_length:
                    end = start + window_size
                    
                    # Extract and reshape windows
                    noisy_windows_list.append(noisy_signal_ch[start:end].reshape(window_size, 1))
                    clean_windows_list.append(clean_signal_ch[start:end].reshape(window_size, 1))
                    
                    start += stride
            
            # Convert to arrays
            noisy_windows_array = np.array(noisy_windows_list)
            clean_windows_array = np.array(clean_windows_list)
            
            # Save the windowed signals
            np.savez_compressed(
                output_file_path,
                noisy_windows=noisy_windows_array,
                clean_windows=clean_windows_array,
                fs=fs
            )
            
            print(f"Processed {noise_type} SNR {snr}: {noisy_windows_array.shape[0]} windows")
        
        except Exception as e:
            print(f"Error processing {noise_type} SNR {snr}: {str(e)}")

print("Windowing complete.")

# After all windowing is done, compile the datasets
print("\nCompiling datasets for easier model training...")
# Create datasets separated by noise type
compile_datasets(separate_noise_types=True)
# Create a combined dataset with all noise types
compile_datasets(separate_noise_types=False)

print("\nAll processing complete!")