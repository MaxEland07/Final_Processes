# ECG Signal Analysis Project

### Overview
This project focuses on analyzing electrocardiogram (ECG) data from the MIT-BIH Arrhythmia Database. The goal is to prepare ECG data for further analysis, which may include classification and denoising techniques.

### Data Source
We use the **MIT-BIH Arrhythmia Database**, a standard dataset for ECG analysis research. This database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects.

### Project Structure
- **Download-Data.py**: Script to download the MIT-BIH Arrhythmia Database
- **Process-Data.py**: Script to preprocess the raw ECG data
- **MIT-BIH Arrhythmia Database/**: Directory containing the dataset
  - **Raw-Data/**: Contains the downloaded raw files
  - **Processed-Data/**: Contains the processed data files
- **Visulisations/**: Directory for visualization scripts and outputs

### Workflow

#### 1. Data Acquisition
The `Download-Data.py` script:
- Uses the `wfdb` library to download all records from the MIT-BIH Arrhythmia dataset
- Saves the raw data files (`.dat`, `.hea`, and `.atr` files) in the `Raw-Data` directory

Each record consists of:
- **`.dat`**: Contains the digitized ECG signal data
- **`.hea`**: Header file with metadata about the record
- **`.atr`**: Annotation file containing beat labels and their locations

#### 2. Data Preprocessing
The `Process-Data.py` script:
- Reads both ECG signal channels and their annotations using the `wfdb` library
- Processes each record to extract the full signal data and corresponding annotations
- Saves each full record as a `.npz` file in the `Processed-Data` directory

Each `.npz` file contains:
- **`signals`**: The complete ECG signal data (typically with shape [n_samples, 2] for two channels)
- **`sample_indices`**: Indices where annotations occur in the signal
- **`labels`**: Heartbeat annotations/labels at those indices
- **`fs`**: The sampling frequency of the signal (typically 360 Hz)

 image.png

### Future Work
Potential next steps for this project include:
- Segmenting the data into individual heartbeats
- Introducing artificial noise at specific SNR levels
- Implementing and evaluating denoising techniques
- Developing classification models for ECG analysis

## Noise 

- Need to add varying levels of noise at specific SNRs to each record.
-- Need to make these accurate to real world 
- Need a way to structure project
- Need a way to have many denosing techniques run
- Need to Calculate increase in SNR over different noises and combinations of noises 
-Record results in table


### Types of noise 

- Baseline wander
- Power line
- Muscle artifact
- Gaussian 

### Measuring improvement 

- SNR
- MSE

Okay so the first thing to do is make some noisy data: 

I want specified SNR's of -6dB, 0dB, 6dB, 12dB, 18dB, and 24dB. 
