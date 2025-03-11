import os
import numpy as np
import wfdb

base_dir = "./MIT-BIH Arrhythmia Database"
raw_dir = os.path.join(base_dir, "Raw-Data")
processed_dir = os.path.join(base_dir, "Processed-Data")

def preprocess_data():
    print("Preprocessing MIT-BIH data into full records...")
    
    # Create the processed_dir if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created directory: {processed_dir}")
    
    records = [f.split('.')[0] for f in os.listdir(raw_dir) if f.endswith('.hea')]
    if not records:
        print("No records found in Raw-Data directory. Please run the download script first.")
        return
    
    # Delete existing files in the processed directory
    for file in os.listdir(processed_dir):
        file_path = os.path.join(processed_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    for record in records:
        try:
            record_path = os.path.join(raw_dir, record)
            data = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')
            
            signals = data.p_signal
            fs = data.fs
            sample_indices = ann.sample
            labels = ann.symbol
            
            output_file = os.path.join(processed_dir, f"{record}_full.npz")
            np.savez(output_file,
                     signals=signals,
                     sample_indices=sample_indices,
                     labels=labels,
                     fs=fs)
            
            print(f"Processed {record}: Full record saved to {output_file} (shape: {signals.shape})")
            
        except Exception as e:
            print(f"Error processing {record}: {e}")

if __name__ == "__main__":
    preprocess_data()
    print(f"Preprocessing Complete! Files saved in {processed_dir}")