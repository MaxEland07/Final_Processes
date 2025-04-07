import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
import multiprocessing
import time

# Define directories
base_dir = "./MIT-BIH Arrhythmia Database"
raw_dir = os.path.join(base_dir, "Raw-Data")
processed_dir = os.path.join(base_dir, "Processed-Data")

# Define filter design functions
def butter_highpass(cutoff, fs, order=5):
    """Design a high-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    """Design a low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, b, a):
    """Apply the designed filter to the data."""
    return filtfilt(b, a, data, axis=0)

def process_record(record_id, channel=0):
    """
    Process a single record with filtering.
    
    Args:
        record_id: The ID of the record to process
        channel: The channel to extract (default: 0, which is MLII in MIT-BIH)
    """
    record_path = os.path.join(raw_dir, record_id)
    output_file = os.path.join(processed_dir, f"{record_id}_full.npz")
    
    try:
        # Load the record
        data = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')

        signals = data.p_signal
        fs = data.fs
        sample_indices = ann.sample
        labels = ann.symbol

        # Extract the specified channel
        if signals.shape[1] <= channel:
            return (record_id, False, f"Channel {channel} not available (only {signals.shape[1]} channels present)")
        
        # Extract single channel and reshape to maintain 2D shape (samples, 1)
        channel_signal = signals[:, channel].reshape(-1, 1)
        
        # Design filters
        highpass_cutoff = 0.5  # Hz (to remove baseline wander)
        lowpass_cutoff = 50    # Hz (to remove high-frequency noise)
        hp_b, hp_a = butter_highpass(highpass_cutoff, fs)
        lp_b, lp_a = butter_lowpass(lowpass_cutoff, fs)

        # Apply high-pass filter (baseline wander removal)
        filtered_signals = apply_filter(channel_signal, hp_b, hp_a)

        # Apply low-pass filter (high-frequency noise removal)
        filtered_signals = apply_filter(filtered_signals, lp_b, lp_a)

        # Save preprocessed signals to file
        np.savez_compressed(output_file,
                signals=filtered_signals,
                sample_indices=sample_indices,
                labels=labels,
                fs=fs,
                channel_name=data.sig_name[channel])

        return (record_id, True, filtered_signals.shape, data.sig_name[channel])
        
    except Exception as e:
        return (record_id, False, str(e))

def preprocess_all_data(use_parallel=True, max_workers=None, channel=0):
    """
    Process all available MIT-BIH records, optionally in parallel.
    
    Args:
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        channel: Which channel to extract (default 0 - MLII lead in MIT-BIH)
    """
    start_time = time.time()
    print(f"Preprocessing all MIT-BIH records (using channel {channel} only)...")

    # Create the processed directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created directory: {processed_dir}")

    # Find all available records in the raw directory
    dat_files = [f for f in os.listdir(raw_dir) if f.endswith('.dat')]
    record_ids = sorted(list(set([f.split('.')[0] for f in dat_files])))
    
    if not record_ids:
        print("No record files found in the raw directory. Please download the data first.")
        return
    
    print(f"Found {len(record_ids)} records: {', '.join(record_ids)}")
    
    # Use CPU count for max workers if not specified
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    successful = 0
    failed = 0
    
    if use_parallel and len(record_ids) > 1:
        print(f"Using parallel processing with {max_workers} workers")
        
        # Process records in parallel
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.starmap(process_record, [(record_id, channel) for record_id in record_ids])
            
        # Process results
        for result in results:
            record_id, success = result[0], result[1]
            if success:
                shape, channel_name = result[2], result[3]
                print(f"Processed {record_id}: Channel {channel} ({channel_name}) saved (shape: {shape})")
                successful += 1
            else:
                print(f"Error processing {record_id}: {result[2]}")
                failed += 1
    else:
        print("Using sequential processing")
        
        # Process each record sequentially
        for record_id in record_ids:
            print(f"\nProcessing record {record_id}...")
            result = process_record(record_id, channel)
            if result[1]:  # success
                shape, channel_name = result[2], result[3]
                print(f"Processed {record_id}: Channel {channel} ({channel_name}) saved (shape: {shape})")
                successful += 1
            else:
                print(f"Error processing {record_id}: {result[2]}")
                failed += 1
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"\nPreprocessing Complete! Total time: {total_time:.2f} seconds")
    print(f"Successfully processed: {successful} records")
    print(f"Failed to process: {failed} records")
    print(f"Files saved in: {processed_dir}")
    print(f"Only channel {channel} was used for all records.")

if __name__ == "__main__":
    # MLII is typically channel 0 in MIT-BIH records
    preprocess_all_data(use_parallel=True, channel=0)