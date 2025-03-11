# Overview

We are preparing a report that evaluates the performance of electrocardiogram (ECG) classification techniques under varying signal-to-noise ratio (SNR) conditions. This report will be valuable for hospitals seeking to determine the most suitable ECG techniques for their needs, as well as for commercial applications, such as smartwatches, where noisy signals are common due to user movement.

The report will include a dedicated section on ECG denoising techniques, exploring both classical signal processing methods and machine learning approaches.

### Methodology

The first step is to generate data for the study. We will use the MIT-BIH database as our source of ECG signals. To simulate real-world conditions, we will introduce artificial noise to these signals at specific SNR levels.

### Objectives

This approach serves two key purposes:

1. **Denoising Evaluation**: It allows us to assess the effectiveness of each preprocessing (denoising) technique across different types of noise and combinations of noise.
2. **Classification Input**: The noisy data will be used as input for testing the performance of various ECG classification methods.

# Data-Production