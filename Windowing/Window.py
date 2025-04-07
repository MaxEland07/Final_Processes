import os
import numpy as np
import multiprocessing
import time
# import wfdb # Not used in this specific snippet, but likely needed elsewhere

# --- Configuration ---
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data") # Path to clean data
noisy_dir = os.path.join(base_dir, "Noisy-Data") # Path to noisy data (adjust if needed)
compiled_dir = os.path.join(base_dir, "Compiled-Data")  # Directory for compiled datasets

# --- Create Output Directories ---
if not os.path.exists(compiled_dir):
    os.makedirs(compiled_dir, exist_ok=True)
    print(f"Created directory: {compiled_dir}")

# Configuration parameters
noise_types = ["ma", "bw", "em"]
snrs = [-12, -6, 0, 6, 12]  # Updated SNR values
window_size = 512
stride = 512

def process_record_windows(record_id, noise_types, snrs, normalize=True):
    """Create windowed data for a single record with all noise types and SNRs."""
    noisy_windows_list = []
    clean_windows_list = []
    metadata_list = []
    
    for noise_type in noise_types:
        for snr in snrs:
            noisy_file_name = f"{record_id}_full_{noise_type}_snr{snr}.npz"
            noisy_file_path = os.path.join(noisy_dir, noisy_file_name)
            
            # Skip if file doesn't exist
            if not os.path.exists(noisy_file_path):
                continue

            try:
                # Load data
                noisy_data = np.load(noisy_file_path)
                
                # Handle different file formats
                if 'noisy_signals' in noisy_data and 'clean_signals' in noisy_data:
                    noisy_signals_full = noisy_data['noisy_signals']
                    clean_signals_full = noisy_data['clean_signals']
                else:
                    # Legacy format compatibility
                    noisy_signals_full = noisy_data['signals']
                    
                    # Load clean data
                    clean_file_name = f"{record_id}_full.npz"
                    clean_file_path = os.path.join(processed_dir, clean_file_name)
                    
                    if not os.path.exists(clean_file_path):
                        continue
                        
                    clean_data = np.load(clean_file_path)
                    clean_signals_full = clean_data['signals']
                
                fs = noisy_data['fs']

                # Verify signal shapes match
                if noisy_signals_full.shape != clean_signals_full.shape:
                    continue
                
                signal_length = noisy_signals_full.shape[0]
                
                # Since we're using single-channel approach, extract the first channel
                # or verify we have a single channel (shape should be (samples, 1))
                if noisy_signals_full.shape[1] > 1:
                    print(f"Warning: File contains {noisy_signals_full.shape[1]} channels, using only the first channel")
                    noisy_signal = noisy_signals_full[:, 0]
                    clean_signal = clean_signals_full[:, 0]
                else:
                    noisy_signal = noisy_signals_full[:, 0]
                    clean_signal = clean_signals_full[:, 0]
                
                # Create sliding windows
                start = 0
                while start + window_size <= signal_length:
                    end = start + window_size
                    
                    noisy_win = noisy_signal[start:end].reshape(window_size, 1)
                    clean_win = clean_signal[start:end].reshape(window_size, 1)
                    
                    # Normalize each window if requested
                    if normalize:
                        # Compute normalization parameters from clean window
                        p_min, p_max = np.percentile(clean_win, [1, 99])
                        win_range = p_max - p_min
                        
                        if win_range > 0:
                            # Normalize both clean and noisy using same parameters
                            win_mean = (p_max + p_min) / 2
                            noisy_win = 2 * (noisy_win - win_mean) / win_range
                            clean_win = 2 * (clean_win - win_mean) / win_range
                    
                    noisy_windows_list.append(noisy_win)
                    clean_windows_list.append(clean_win)
                    metadata_list.append({
                        'record_id': record_id,
                        'noise_type': noise_type,
                        'snr': snr
                    })
                    
                    start += stride
                
                print(f"Processed {record_id} {noise_type} SNR {snr}")
            
            except Exception as e:
                print(f"Error processing {record_id} {noise_type} SNR {snr}: {str(e)}")
    
    return noisy_windows_list, clean_windows_list, metadata_list

def create_unified_dataset(train_split=0.8, test_split=0.2, normalize=True, use_parallel=True, max_workers=None):
    """Create a unified train/test dataset directly from the noisy records."""
    # Access configuration parameters
    global noise_types, snrs, window_size
    noise_types_config = noise_types
    snrs_config = snrs
    
    start_time = time.time()
    
    # Find all available noisy records
    noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith('.npz')]
    record_ids = sorted(list(set([f.split('_')[0] for f in noisy_files])))
    
    if not record_ids:
        print("No noisy records found. Please run Generate-Noise.py first.")
        return
    
    print(f"Found {len(record_ids)} records with noisy data")
    print(f"Using SNR values: {snrs_config}")
    print(f"Using noise types: {noise_types_config}")
    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Single-channel processing mode")
    
    # Use CPU count for max workers if not specified
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    all_noisy_windows = []
    all_clean_windows = []
    all_metadata = []
    
    # Prepare arguments for each record
    args = [(record_id, noise_types_config, snrs_config, normalize) for record_id in record_ids]
    
    if use_parallel and len(record_ids) > 1:
        print(f"Using parallel processing with {max_workers} workers")
        
        # Process records in parallel
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.starmap(process_record_windows, args)
            
        # Process results
        for noisy_windows, clean_windows, metadata in results:
            if noisy_windows:
                all_noisy_windows.extend(noisy_windows)
                all_clean_windows.extend(clean_windows)
                all_metadata.extend(metadata)
    else:
        print("Using sequential processing")
        
        # Process each record sequentially
        for arg_tuple in args:
            record_id, noise_types_local, snrs_local, normalize_local = arg_tuple
            print(f"Processing record {record_id}...")
            noisy_windows, clean_windows, metadata = process_record_windows(record_id, noise_types_local, snrs_local, normalize_local)
            if noisy_windows:
                all_noisy_windows.extend(noisy_windows)
                all_clean_windows.extend(clean_windows)
                all_metadata.extend(metadata)
    
    if not all_noisy_windows:
        print("No valid windows found.")
        return
    
    # Convert to numpy arrays
    all_noisy = np.array(all_noisy_windows)
    all_clean = np.array(all_clean_windows)
    all_metadata = np.array(all_metadata, dtype=object)
    
    # Shuffle the data with fixed seed for reproducibility
    n_windows = len(all_noisy)
    indices = np.arange(n_windows)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    all_noisy = all_noisy[indices]
    all_clean = all_clean[indices]
    all_metadata = all_metadata[indices]
    
    # Split into train/test sets
    split_idx = int(n_windows * train_split)
    
    train_noisy = all_noisy[:split_idx]
    train_clean = all_clean[:split_idx]
    train_metadata = all_metadata[:split_idx]
    
    test_noisy = all_noisy[split_idx:]
    test_clean = all_clean[split_idx:]
    test_metadata = all_metadata[split_idx:]
    
    # Save as a single train/test dataset
    print(f"Saving unified dataset with {n_windows} total windows")
    
    # Create compiled directory if it doesn't exist
    if not os.path.exists(compiled_dir):
        os.makedirs(compiled_dir, exist_ok=True)
    
    np.savez_compressed(
        os.path.join(compiled_dir, "ecg_denoising_train.npz"),
        noisy=train_noisy,
        clean=train_clean,
        metadata=train_metadata,
        normalized=normalize
    )
    
    np.savez_compressed(
        os.path.join(compiled_dir, "ecg_denoising_test.npz"),
        noisy=test_noisy,
        clean=test_clean,
        metadata=test_metadata,
        normalized=normalize
    )
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"\nData processing complete! Total time: {total_time:.2f} seconds")
    print(f"Saved unified dataset:")
    print(f"  Train: {train_noisy.shape[0]} samples")
    print(f"  Test:  {test_noisy.shape[0]} samples")
    
    # Print summary statistics
    print("\nDataset summary:")
    print(f"  Total windows: {n_windows}")
    print(f"  Window size: {window_size}")
    print(f"  Feature dimension: {train_noisy.shape[2]}")
    print(f"  Normalization: {'Applied' if normalize else 'None'}")
    
    if normalize:
        # Calculate and print statistics about the normalized data
        train_noisy_mean = np.mean(train_noisy)
        train_noisy_std = np.std(train_noisy)
        train_clean_mean = np.mean(train_clean)
        train_clean_std = np.std(train_clean)
        
        print("\nNormalization statistics:")
        print(f"  Train noisy data: mean={train_noisy_mean:.4f}, std={train_noisy_std:.4f}")
        print(f"  Train clean data: mean={train_clean_mean:.4f}, std={train_clean_std:.4f}")
    
    return True

if __name__ == "__main__":
    # Create unified train/test dataset directly
    print("Creating unified train/test dataset...")
    create_unified_dataset(train_split=0.8, test_split=0.2, normalize=True, use_parallel=True)
    
    print("\nAll processing complete!")