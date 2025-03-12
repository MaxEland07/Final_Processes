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

    # Download MIT-BIH Arrhythmia Database (only record 100)
    print("Downloading MIT-BIH Arrhythmia Database (record 100)...")
    try:
        wfdb.dl_database('mitdb', records=['100'], dl_dir=raw_dir)
        print("Record 100 downloaded successfully.")
    except Exception as e:
        print(f"Error downloading record 100: {e}")
        return

    # Download MIT-BIH Noise Stress Test Database
    print("Downloading MIT-BIH Noise Stress Test Database...")
    try:
        wfdb.dl_database('nstdb', dl_dir=stress_test_dir)
    except Exception as e:
        print(f"Error downloading Noise Stress Test Database: {e}")

if __name__ == "__main__":
    download_data()
    print("Data Downloaded Successfully! Files saved in:")
    print(f"- ./MIT-BIH Arrhythmia Database/Raw-Data (Arrhythmia Database, record 100)")
    print(f"- ./MIT-BIH Arrhythmia Database/stress_test (Noise Stress Test Database)")