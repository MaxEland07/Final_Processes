### Overview
We are preparing a report that evaluates the performance of electrocardiogram (ECG) classification techniques under varying signal-to-noise ratio (SNR) conditions. This report will be valuable for hospitals seeking to determine the most suitable ECG techniques for their needs, as well as for commercial applications, such as smartwatches, where noisy signals are common due to user movement.

The report will include a dedicated section on ECG denoising techniques, exploring both classical signal processing methods and machine learning approaches.

### Methodology
The first step is to generate data for the study. We are using the **MIT-BIH Arrhythmia Database** as our source of ECG signals. To simulate real-world conditions, we will introduce artificial noise to these signals at specific SNR levels.

### Objectives
This approach serves two key purposes:

1. **Denoising Evaluation**: It allows us to assess the effectiveness of each preprocessing (denoising) technique across different types of noise and combinations of noise.
2. **Classification Input**: The noisy data will be used as input for testing the performance of various ECG classification methods.

### Data Processing
#### MIT-BIH Arrhythmia Database
We start by downloading all the records from the **MIT-BIH Arrhythmia dataset**. Each record consists of three file types:
- **`.dat`**: Contains the digitized ECG signal data.
- **`.hea`**: Header file with metadata about the record.
- **`.atr`**: Annotation file containing beat labels and their locations.

#### Data Preprocessing
Our preprocessing script performs the following steps:

1. **Loading Raw Data**: 
   - Uses the `wfdb` library to read both ECG signal channels and their annotations.

2. **Segmentation**:
   - Extracts fixed-size segments (default: 512 samples) centered around each annotated heartbeat.
   - **Both ECG channels** are included in each segment.

3. **Data Organization**:
   - Segments from all records are stored as NumPy arrays.
   - Corresponding heartbeat labels are saved alongside the segments.

4. **Storage in `.npz` Files**:
   - Processed data for each record is saved in a separate `.npz` file in the `Processed-Data` directory.
   - Each `.npz` file contains:
     - **`signals`**: A 3D array of shape `(n_segments, window_size, 2)`, where 2 represents the two ECG channels.
     - **`labels`**: A 1D array of shape `(n_segments,)` with heartbeat labels.
     - **`fs`**: The sampling frequency of the signal (typically 360 Hz).

This structured format allows for efficient storage and easy access to the preprocessed ECG data, facilitating subsequent analysis and machine learning tasks.
