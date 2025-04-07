import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define directories
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data")
noise_dir = os.path.join(base_dir, "Noisy-Data")
denoised_dir = os.path.join(base_dir, "Denoised-Data")
results_dir = "./Results"
vis_dir = "./Visulisations/Filtering_Results"

if not os.path.exists(denoised_dir, vis_dir, results_dir):
    os.makedirs(denoised_dir, vis_dir, results_dir, exist_ok=True)


# load windowed data 
windowed_dir = os.path.join(base_dir, "Windowed-Data")
windowed_data = np.load(os.path.join(windowed_dir, "100_MA_0dB_win512_stride256.npz"))

# Get the noisy and clean windows
noisy_windows = windowed_data["noisy_windows"]
clean_windows = windowed_data["clean_windows"]  

# Check the shapes
print(f"Noisy windows shape: {noisy_windows.shape}")
print(f"Clean windows shape: {clean_windows.shape}")

# Build the LSTM-based denoising model
def build_lstm_denoiser(input_shape=(512, 1)):
    """
    Builds the LSTM-based denoising model.

    Args:
        input_shape (tuple): The shape of the input window
                          (window_length, num_features).

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = Sequential()
    # Expects input shape: (batch_size, timesteps, features)
    # timesteps = 512, features = 1 (ECG amplitude)
    model.add(LSTM(140, input_shape=input_shape, return_sequences=True))
    model.add(Dense(140, activation='relu')) # Apply Dense to each time step output
    model.add(LSTM(140, return_sequences=True))
    model.add(Dense(140, activation='relu')) # Apply Dense to each time step output
    # Output layer predicts the denoised value for each time step
    model.add(Dense(1, activation='linear'))

    # Compile the model within the build function or separately before training
    # Example compilation:
    # model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# Compile the model
model = build_lstm_denoiser()
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
# Assuming you have a training dataset with X_train (noisy windows) and y_train (clean windows)
# X_train = noisy_windows
# y_train = clean_windows  
# Split the data into training and validation sets (70-30 split)
split_idx = int(0.7 * len(noisy_windows))
X_train = noisy_windows[:split_idx]
y_train = clean_windows[:split_idx]
X_val = noisy_windows[split_idx:]
y_val = clean_windows[split_idx:]

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))


# Save the trained model
model.save(os.path.join(results_dir, "lstm_denoiser.keras"))






