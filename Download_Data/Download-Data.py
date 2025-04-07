import wfdb
import shutil
import os

# Define Directories
base_dir = "./MIT-BIH Arrhythmia Database"
raw_dir = os.path.join(base_dir, "Raw-Data")
stress_test_dir = os.path.join(base_dir, "stress_test")

# List of known valid records in the MIT-BIH Arrhythmia Database
# This is based on the actual available records in the database
VALID_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124'
    # Note: 110, 120 are missing from this range
]

# Create directories if they don't exist
for directory in [base_dir, raw_dir, stress_test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_data(clear_existing=False, record_range=None):
    """
    Downloads MIT-BIH Arrhythmia Database records and Noise Stress Test Database.
    
    Args:
        clear_existing: Whether to clear existing directories before downloading
        record_range: Range of record numbers to download (default: 100-124)
    """
    # Set default record range if not specified
    if record_range is None:
        record_range = range(100, 125)  # Records 100-124
    
    # Optionally clear existing directories
    if clear_existing:
        for directory in [raw_dir, stress_test_dir]:
            if os.path.exists(directory):
                print(f"Clearing existing directory: {directory}")
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
    
    # Download specified MIT-BIH Arrhythmia Database records
    print(f"Downloading MIT-BIH Arrhythmia Database records {record_range.start}-{record_range.stop-1}...")
    success = True
    downloaded_records = []
    skipped_records = []
    
    # Convert record_range to a filtered list using VALID_RECORDS
    records_to_download = [str(r) for r in record_range if str(r) in VALID_RECORDS]
    
    if not records_to_download:
        print("No valid records in the specified range.")
        success = False
    else:
        try:
            # Download all valid records at once (more efficient)
            print(f"Downloading records: {', '.join(records_to_download)}...")
            wfdb.dl_database('mitdb', dl_dir=raw_dir, records=records_to_download)
            downloaded_records = records_to_download
            print(f"Records downloaded successfully.")
        except Exception as e:
            print(f"Error downloading records as batch: {e}")
            print("Trying individual downloads instead...")
            
            # Individual downloads as fallback
            downloaded_records = []
            for record_id in records_to_download:
                try:
                    print(f"Downloading record {record_id}...")
                    wfdb.dl_database('mitdb', dl_dir=raw_dir, records=[record_id])
                    downloaded_records.append(record_id)
                except Exception as e:
                    print(f"Error downloading record {record_id}: {e}")
                    skipped_records.append(record_id)
                    success = False
    
    # Download MIT-BIH Noise Stress Test Database
    print("\nDownloading MIT-BIH Noise Stress Test Database...")
    try:
        wfdb.dl_database('nstdb', dl_dir=stress_test_dir)
        print("Noise Stress Test Database downloaded successfully.")
    except Exception as e:
        print(f"Error downloading Noise Stress Test Database: {e}")
        success = False

    return success, downloaded_records, skipped_records

if __name__ == "__main__":
    print("Starting download of MIT-BIH databases...")
    
    # Download records 100-124 (will automatically skip missing ones)
    success, downloaded, skipped = download_data(clear_existing=False, record_range=range(100, 125))
    
    if success or downloaded:
        print("\nData Downloaded:")
        print(f"- {raw_dir} (MIT-BIH Arrhythmia Database records)")
        print(f"- {stress_test_dir} (Noise Stress Test Database)")
        
        print(f"\nSuccessfully downloaded {len(downloaded)} records: {', '.join(downloaded)}")
        
        if skipped:
            print(f"Skipped {len(skipped)} unavailable records: {', '.join(skipped)}")
        
        # Verify downloaded files
        arrhythmia_files = [f for f in os.listdir(raw_dir) if f.endswith('.dat')]
        record_numbers = sorted(list(set([f.split('.')[0] for f in arrhythmia_files])))
        
        if record_numbers:
            print(f"\nVerified {len(record_numbers)} records in directory: {', '.join(record_numbers)}")
        else:
            print("\nWarning: No record files found in the output directory.")
    else:
        print("\nDownload process encountered serious errors. Please check the logs above.")