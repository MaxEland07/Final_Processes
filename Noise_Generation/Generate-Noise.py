import os
import numpy as np
import wfdb
import multiprocessing
import time

# Define directories
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data")
noisy_dir = os.path.join(base_dir, "Noisy-Data")
stress_test_dir = os.path.join(base_dir, "stress_test")

# Create noisy_data directory if it doesn't exist
os.makedirs(noisy_dir, exist_ok=True)

# Function to add noise and compute SNR
def add_noise(clean_signal, noise, target_snr):
    """
    Add noise to a clean signal at a specified SNR level.
    
    Args:
        clean_signal: The clean ECG signal
        noise: The noise signal to add
        target_snr: The target Signal-to-Noise Ratio in dB
    
    Returns:
        The signal with added noise at the specified SNR
    """
    # Ensure the noise has the same length as the clean signal
    if len(noise) > len(clean_signal):
        noise = noise[:len(clean_signal)]
    elif len(noise) < len(clean_signal):
        # Pad by repeating
        repeats = int(np.ceil(len(clean_signal) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean_signal)]
    
    # Normalize noise to zero mean (if not already)
    noise = noise - np.mean(noise)
    
    # Calculate power of signal and noise
    clean_rms = np.sqrt(np.mean(clean_signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    
    # Avoid division by zero
    if noise_rms < 1e-10:
        noise_rms = 1e-10
    if clean_rms < 1e-10:
        clean_rms = 1e-10
    
    # Calculate the desired noise level based on SNR
    desired_noise_rms = clean_rms / (10**(target_snr/20))
    
    # Scale the noise
    gain = desired_noise_rms / noise_rms
    scaled_noise = noise * gain
    
    # Add the scaled noise to the signal
    noisy_signal = clean_signal + scaled_noise
    
    # Verify actual SNR (for debugging)
    actual_snr = 20 * np.log10(clean_rms / np.sqrt(np.mean(scaled_noise**2)))
    print(f"Target SNR: {target_snr} dB, Achieved SNR: {actual_snr:.2f} dB")
    
    return noisy_signal

def generate_noisy_record(record_id, noise_types, snr_values):
    """
    Process a single record, adding different types of noise at different SNR levels.
    Works with single-channel ECG data.
    """
    try:
        # Load processed data
        input_file = os.path.join(processed_dir, f"{record_id}_full.npz")
        if not os.path.exists(input_file):
            return (record_id, False, f"Processed file not found: {input_file}")
            
        data = np.load(input_file)
        filtered_signals = data['signals']  # Shape: (samples, 1) for single channel
        fs = data['fs']  # Sampling frequency
        
        # Include other metadata if available
        metadata = {}
        for key in data.files:
            if key != 'signals':
                metadata[key] = data[key]
        
        # Verify shape - ensure we're working with a single channel
        if filtered_signals.ndim == 1:
            # If 1D array, reshape to 2D (samples, 1)
            filtered_signals = filtered_signals.reshape(-1, 1)
        elif filtered_signals.shape[1] > 1:
            # If multiple channels, extract only the first channel
            print(f"Warning: File contains {filtered_signals.shape[1]} channels, using only the first channel")
            filtered_signals = filtered_signals[:, 0:1]  # Keep as 2D: (samples, 1)
        
        # Get signal length
        signal_length = filtered_signals.shape[0]
        
        # Process each noise type
        files_generated = 0
        
        for noise_type in noise_types:
            try:
                # Ensure we're using the correct approach to load the noise data
                noise_path = os.path.join(stress_test_dir, noise_type)
                
                # Load noise samples using wfdb (most reliable method)
                try:
                    print(f"Loading noise {noise_type} using wfdb...")
                    noise_record = wfdb.rdrecord(noise_path, channels=[0])
                    noise_data = noise_record.p_signal.flatten()
                    print(f"Successfully loaded noise: {len(noise_data)} samples, sampling rate: {noise_record.fs} Hz")
                    
                    # Handle potential sampling rate differences
                    if hasattr(noise_record, 'fs') and noise_record.fs != fs:
                        print(f"Warning: Noise sampling rate ({noise_record.fs} Hz) differs from signal ({fs} Hz)")
                        # Simple resampling could be added here if needed
                except Exception as e:
                    print(f"Error loading noise with wfdb: {e}")
                    print("Attempting to load as binary file...")
                    
                    # Fallback to direct file read
                    noise_file = f"{noise_path}.dat"
                    if os.path.exists(noise_file):
                        noise_data = np.fromfile(noise_file, dtype=np.int16)
                        print(f"Loaded noise from binary file: {len(noise_data)} samples")
                    else:
                        return (record_id, False, f"Could not find noise file for {noise_type}")
                
                # Verify we have valid noise data
                if noise_data.size == 0:
                    return (record_id, False, f"Empty noise data for {noise_type}")
                    
                # Set seed for reproducibility
                seed = int(record_id) if record_id.isdigit() else hash(record_id) % 10000
                np.random.seed(seed)
                
                # Select a random starting position if noise is long enough
                if len(noise_data) > signal_length:
                    start_pos = np.random.randint(0, len(noise_data) - signal_length)
                    noise = noise_data[start_pos:start_pos + signal_length]
                else:
                    # If noise is shorter, repeat it
                    repeats = int(np.ceil(signal_length / len(noise_data)))
                    noise = np.tile(noise_data, repeats)[:signal_length]
                
                # Process each SNR value
                for snr_value in snr_values:
                    clean_channel = filtered_signals[:, 0]
                    try:
                        noisy_channel = add_noise(clean_channel, noise, snr_value)
                        # Reshape to (samples, 1) to maintain 2D structure
                        noisy_signals = noisy_channel.reshape(-1, 1)
                    except Exception as e:
                        return (record_id, False, f"Error adding noise: {str(e)}")
                    
                    # Save results
                    output_file = os.path.join(noisy_dir, f"{record_id}_full_{noise_type}_snr{snr_value}.npz")
                    
                    # Combine noise data with metadata
                    save_data = {
                        'noisy_signals': noisy_signals,
                        'clean_signals': filtered_signals,
                        'fs': fs,
                        'noise_type': noise_type,
                        'snr': snr_value
                    }
                    
                    # Add back any additional metadata from original file
                    for key, value in metadata.items():
                        if key not in save_data:
                            save_data[key] = value
                    
                    np.savez_compressed(output_file, **save_data)
                    files_generated += 1
                    
            except Exception as e:
                return (record_id, False, f"Error with noise type {noise_type}: {str(e)}")
        
        return (record_id, True, f"Generated {files_generated} noisy files")
        
    except Exception as e:
        return (record_id, False, f"General error: {str(e)}")

def generate_all_noisy_data(use_parallel=True, max_workers=None):
    """
    Generate noisy versions of all available records.
    
    Args:
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of worker processes to use
    """
    start_time = time.time()
    print("Generating noisy data for all records...")
    
    # Define noise types and SNR values
    noise_types = ['bw', 'ma', 'em']  # baseline wander, muscle artifact, electrode motion
    snr_values = [-12, -6, 0, 6, 12]  # dB - updated per requirements
    
    # Find all processed record files
    if not os.path.exists(processed_dir):
        print(f"Error: Processed directory not found: {processed_dir}")
        return
        
    npz_files = [f for f in os.listdir(processed_dir) if f.endswith('_full.npz')]
    record_ids = [f.split('_')[0] for f in npz_files]
    
    if not record_ids:
        print("No processed records found. Please run Process-Data.py first.")
        return
    
    print(f"Found {len(record_ids)} processed records: {', '.join(record_ids)}")
    print(f"Will apply noise types: {', '.join(noise_types)}")
    print(f"SNR values: {snr_values} dB")
    print("Using only the first channel from each record")
    
    # Use CPU count for max workers if not specified
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Prepare arguments for each record
    args = [(record_id, noise_types, snr_values) for record_id in record_ids]
    
    successful = 0
    failed = 0
    
    if use_parallel and len(record_ids) > 1:
        print(f"Using parallel processing with {max_workers} workers")
        
        # Process records in parallel
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.starmap(generate_noisy_record, args)
            
        # Process results
        for record_id, success, message in results:
            if success:
                print(f"Record {record_id}: {message}")
                successful += 1
            else:
                print(f"Record {record_id}: Failed - {message}")
                failed += 1
    else:
        print("Using sequential processing")
        
        # Process each record sequentially
        for record_id, noise_types, snr_values in args:
            print(f"\nProcessing record {record_id}...")
            result = generate_noisy_record(record_id, noise_types, snr_values)
            if result[1]:  # success
                print(f"Record {record_id}: {result[2]}")
                successful += 1
            else:
                print(f"Record {record_id}: Failed - {result[2]}")
                failed += 1
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"\nNoise generation complete! Total time: {total_time:.2f} seconds")
    print(f"Successfully processed: {successful} records")
    print(f"Failed to process: {failed} records")
    print(f"All noisy files saved in {noisy_dir}")
    print(f"Generated files at SNRs: {snr_values} dB")
    print("All files contain single-channel ECG data with added noise")

def check_noise_files():
    """Debug function to examine the noise files"""
    print("\nChecking noise files in the stress_test directory:")
    
    files = os.listdir(stress_test_dir)
    print(f"Files in directory: {files}")
    
    # Test adding noise directly with a simple signal
    for noise_type in ['bw', 'ma', 'em']:
        try:
            noise_path = os.path.join(stress_test_dir, noise_type)
            print(f"\nTesting {noise_type}:")
            
            # Load noise
            noise_sig, fields = wfdb.rdsamp(noise_path, channels=[0])
            noise_data = noise_sig.flatten()
            print(f"Loaded noise: {len(noise_data)} samples")
            
            # Create a simple test signal
            test_signal = np.sin(np.linspace(0, 100*np.pi, 1000))
            
            # Try adding noise
            snr = 0  # 0dB SNR
            test_noisy = add_noise(test_signal, noise_data[:1000], snr)
            
            print(f"Successfully added noise: input={test_signal.shape}, output={test_noisy.shape}")
            print(f"  Original signal mean: {np.mean(test_signal**2)}")
            print(f"  Noisy signal mean: {np.mean(test_noisy**2)}")
            
        except Exception as e:
            print(f"Error testing {noise_type}: {str(e)}")

if __name__ == "__main__":
    # Run the full noise generation process
    generate_all_noisy_data(use_parallel=True)