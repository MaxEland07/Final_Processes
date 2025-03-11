import wfdb 
import shutil
import os

#Define Directories
base_dir = "./MIT-BIH Arrhythmia Database"
raw_dir = os.path.join(base_dir, "Raw-Data")

# Clear existing directories and create new ones
for directory in [base_dir, raw_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def download_data():
    #Download Data
    print("Downloading Data...")

    records = wfdb.get_record_list('mitdb')
    if not records:
        print("No records found")
        return
    
    for record in records:
        print(f"Downloading {record}...")
        try:
            wfdb.dl_database('mitdb', records=[record], dl_dir=raw_dir) 
        except Exception as e:
            print(f"Error downloading {record}: {e}")


if __name__ == "__main__":
    download_data()
    print("Data Downloaded Successfully! Files saved in ./MIT-BIH Arrhythmia Database/Raw-Data")

