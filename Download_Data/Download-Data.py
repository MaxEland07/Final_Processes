import wfdb
import shutil
import os

# Define Directories
base_dir = "./MIT-BIH Arrhythmia Database"
raw_dir = os.path.join(base_dir, "Raw-Data")
stress_test_dir = os.path.join(base_dir, "stress_test")

# Clear existing directories and create new ones
for directory in [base_dir, raw_dir, stress_test_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def download_data():
    # Download Data
    print("Downloading Data...")

    # Download MIT-BIH Arrhythmia Database
    print("Downloading MIT-BIH Arrhythmia Database...")
    records = wfdb.get_record_list('mitdb')
    if not records:
        print("No records found for Arrhythmia Database")
        return
    
    for record in records:
        print(f"Downloading {record}...")
        try:
            wfdb.dl_database('mitdb', records=[record], dl_dir=raw_dir)
        except Exception as e:
            print(f"Error downloading {record}: {e}")

    # Download MIT-BIH Noise Stress Test Database
    print("Downloading MIT-BIH Noise Stress Test Database...")
    try:
        wfdb.dl_database('nstdb', dl_dir=stress_test_dir)
    except Exception as e:
        print(f"Error downloading Noise Stress Test Database: {e}")

if __name__ == "__main__":
    download_data()
    print("Data Downloaded Successfully! Files saved in:")
    print(f"- ./MIT-BIH Arrhythmia Database/Raw-Data (Arrhythmia Database)")
    print(f"- ./MIT-BIH Arrhythmia Database/stress_test (Noise Stress Test Database)")