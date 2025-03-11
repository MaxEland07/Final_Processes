# ECG Signal Analysis and Denoising Project

## Overview
This project focuses on evaluating the performance of ECG denoising techniques under various noise conditions. We process ECG data from the MIT-BIH Arrhythmia Database, add synthetic noise at different signal-to-noise ratio (SNR) levels, and then evaluate various filtering methods to remove this noise.

## Project Structure
- **Download-Data.py**: Script to download the MIT-BIH Arrhythmia Database
- **Process-Data.py**: Script to preprocess the raw ECG data with basic filters
- **Noise_Generation/Generate-Noise.py**: Script to add different types of noise to the ECG signals
- **Denoising/Filter_Design.py**: Script to apply various denoising techniques and evaluate performance
- **MIT-BIH Arrhythmia Database/**: Directory containing:
  - **Raw-Data/**: Original downloaded database files
  - **Processed-Data/**: Cleaned and preprocessed ECG signals
  - **Noise-Data/**: ECG signals with synthetic noise added
  - **Denoised-Data/**: ECG signals after applying denoising techniques
- **Results/**: Directory containing performance metrics of denoising techniques
- **Visulisations/**: Directory containing visualizations and comparisons

## Workflow

### 1. Data Acquisition
The `Download-Data.py` script:
- Uses the `wfdb` library to download records from the MIT-BIH Arrhythmia dataset
- Saves the raw data files (`.dat`, `.hea`, and `.atr`) in the `Raw-Data` directory

### 2. Data Preprocessing
The `Process-Data.py` script:
- Reads ECG signal channels and their annotations
- Applies basic filtering to remove baseline wander (high-pass filter at 0.5 Hz)
- Removes high-frequency noise (low-pass filter at 50 Hz)
- Saves each record as a `.npz` file in the `Processed-Data` directory

### 3. Noise Generation
The `Noise_Generation/Generate-Noise.py` script:
- Adds synthetic noise to the preprocessed ECG signals at various SNR levels (-6 to 24 dB)
- Implements multiple noise types that mimic real-world ECG recording conditions:
  - **Gaussian Noise (G)**: Random noise following normal distribution
  - **Baseline Wander (BW)**: Low-frequency oscillations simulating respiration and body movement
  - **Power Line Interference (PL)**: 50/60 Hz interference from power supplies
  - **Muscle Artifact (MA)**: High-frequency noise simulating EMG contamination
  - **Electrode Motion (EM)**: Abrupt changes mimicking electrode movement
- Creates combinations of these noise types to simulate complex real-world scenarios
- Saves noisy signals in the `Noise-Data` directory, organized by noise type and SNR level

### 4. Signal Denoising
The `Denoising/Filter_Design.py` script:
- Implements multiple filtering techniques:
  - **Butterworth Bandpass**: Classical approach for removing noise outside specific frequency bands
  - **Savitzky-Golay**: Smoothing filter that preserves signal features
  - **Wavelet Denoising**: Multi-resolution approach that separates signal from noise
  - **Moving Average**: Simple smoothing filter for comparison
- Applies these techniques to the noisy signals
- Calculates SNR before and after filtering to measure performance
- Saves denoised signals in the `Denoised-Data` directory
- Generates visualizations comparing original, noisy, and denoised signals
- Creates performance summaries comparing filter effectiveness across noise types and SNR levels

## Results
The filtering results are summarized in:
- **filter_performance.csv**: Comprehensive metrics on each filter's performance
- **Visualization plots**: Visual comparisons of original, noisy, and denoised signals
- **Summary charts**: Performance comparisons across noise types and SNR levels

## Conclusion
This project provides a framework for:
1. Evaluating the robustness of ECG denoising techniques under various noise conditions
2. Identifying optimal filtering approaches for specific noise types
3. Understanding the relationship between SNR levels and filter performance
4. Establishing best practices for ECG signal preprocessing in both clinical and wearable device applications

## Noise 

- Need to add varying levels of noise at specific SNRs to each record.
-- Need to make these accurate to real world 
-- https://ime.um.edu.mo/wp-content/uploads/presentations/8555ab0da8e657ce9ef794577f1e1bd9.pdf
-- Gives 
- Need a way to structure project
- Need a way to have many denosing techniques run
- Need to Calculate increase in SNR over different noises and combinations of noises 
-Record results in table


### Types of noise 

- Baseline wander
- Power line
- Muscle artifact
- Gaussian 
- Electrode Motion
- BW + MA
- PL + G
- BW + PL
- ALL

### Measuring improvement 

- SNR
- MSE

Okay so the first thing to do is make some noisy data: 

I want specified SNR's of -6dB, 0dB, 6dB, 12dB, 18dB, and 24dB. 
