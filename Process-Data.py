import wfdb
import os
import numpy as np

# Define Directories
base_dir = "./MIT-BIH Arrhythmia Database"
raw_dir = os.path.join(base_dir, "Raw-Data")
processed_dir = os.path.join(base_dir, "Processed-Data")

# Create processed directory if it doesn't exist
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

def preprocess_data(window_size=512):
    """
    Preprocess MIT-BIH raw files into segmented NumPy arrays with annotations.
    
    Parameters:
    - window_size: Number of samples per segment (default: 512).
    """
    print("Preprocessing MIT-BIH data into usable NumPy arrays...")
    
    # Get list of records from the raw directory
    records = [f.split('.')[0] for f in os.listdir(raw_dir) if f.endswith('.hea')]
    if not records:
        print("No records found in Raw-Data directory. Please run the download script first.")
        return
    
    for record in records:
        try:
            # Load signal and annotations
            record_path = os.path.join(raw_dir, record)
            data = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')
            
            # Extract signals and metadata
            signals = data.p_signal  # Shape: (n_samples, 2) - two channels
            fs = data.fs  # Sampling frequency (typically 360 Hz)
            sample_indices = ann.sample  # Annotation sample indices
            labels = ann.symbol  # Annotation labels (e.g., 'N', 'V', 'A')
            
            # Segment signals around each annotation
            segments = []
            segment_labels = []
            half_window = window_size // 2
            
            for idx, label in zip(sample_indices, labels):
                # Define window boundaries centered on annotation
                start = idx - half_window
                end = idx + half_window
                
                # Skip if window exceeds signal bounds
                if start < 0 or end > len(signals):
                    continue
                
                # Extract segment from both channels
                segment = signals[start:end, :]  # Shape: (window_size, 2) - both channels included
                segments.append(segment)
                segment_labels.append(label)
            
            # Convert lists to NumPy arrays
            segments_array = np.array(segments)  # Shape: (n_segments, window_size, 2)
            labels_array = np.array(segment_labels)  # Shape: (n_segments,)
            
            # Save as .npz file
            output_file = os.path.join(processed_dir, f"{record}_segments.npz")
            np.savez(output_file, 
                     signals=segments_array, 
                     labels=labels_array, 
                     fs=fs)
            
            print(f"Processed {record}: {len(segments)} segments saved to {output_file} "
                  f"(shape: {segments_array.shape})")
            
        except Exception as e:
            print(f"Error processing {record}: {e}")

if __name__ == "__main__":
    # Preprocess the data
    preprocess_data(window_size=512)
    print(f"Preprocessing Complete! Files saved in {processed_dir}")
