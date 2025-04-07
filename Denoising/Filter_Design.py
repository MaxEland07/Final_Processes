import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define directories
base_dir = "./MIT-BIH Arrhythmia Database"
processed_dir = os.path.join(base_dir, "Processed-Data")
compiled_dir = os.path.join(base_dir, "Compiled-Data")
results_dir = "./Results"
vis_dir = "./Visulisations/Filtering_Results"

# Create directories if they don't exist
for directory in [results_dir, vis_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Configuration parameters
record_id = "100"
noise_types = ["ma", "bw", "em"]
training_mode = "combined"  # Options: "combined", "separate", or "specific"
specific_noise = "ma"       # Only used if training_mode is "specific"

# Build the LSTM-based denoising model
def build_lstm_denoiser(input_shape=(512, 1)):
    """
    Builds the LSTM-based denoising model.
    """
    model = Sequential()
    model.add(LSTM(140, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(140, activation='relu'))
    model.add(LSTM(140, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(140, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

# Training function
def train_model(train_data, valid_data, model_name, epochs=50):
    # Load data
    X_train = train_data["noisy"]
    y_train = train_data["clean"]
    X_val = valid_data["noisy"] 
    y_val = valid_data["clean"]
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    
    # Compile the model
    model = build_lstm_denoiser()
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        ModelCheckpoint(
            os.path.join(results_dir, f"{model_name}_best.keras"),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train, 
        epochs=epochs,
        batch_size=32, 
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(results_dir, f"{model_name}.keras"))
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Denoiser Training: {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(os.path.join(vis_dir, f"training_history_{model_name}.png"))
    plt.close()
    
    # Evaluate on validation set
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation MSE: {val_loss:.6f}")
    
    return model, history

# Main training workflow
try:
    if training_mode == "combined":
        # Train a single model on all noise types combined
        print("Training combined model for all noise types...")
        train_file = os.path.join(compiled_dir, f"{record_id}_all_noise_types_train.npz")
        valid_file = os.path.join(compiled_dir, f"{record_id}_all_noise_types_valid.npz")
        
        train_data = np.load(train_file)
        valid_data = np.load(valid_file)
        
        model, history = train_model(
            train_data, 
            valid_data, 
            model_name="lstm_denoiser_all_noise_types"
        )
        
    elif training_mode == "separate":
        # Train separate models for each noise type
        for noise_type in noise_types:
            print(f"\nTraining model for {noise_type.upper()} noise...")
            train_file = os.path.join(compiled_dir, f"{record_id}_{noise_type}_train.npz")
            valid_file = os.path.join(compiled_dir, f"{record_id}_{noise_type}_valid.npz")
            
            if not os.path.exists(train_file) or not os.path.exists(valid_file):
                print(f"Files for {noise_type} not found, skipping...")
                continue
                
            train_data = np.load(train_file)
            valid_data = np.load(valid_file)
            
            model, history = train_model(
                train_data, 
                valid_data, 
                model_name=f"lstm_denoiser_{noise_type}"
            )
            
    elif training_mode == "specific":
        # Train on a specific noise type
        noise_type = specific_noise
        print(f"\nTraining model for {noise_type.upper()} noise...")
        train_file = os.path.join(compiled_dir, f"{record_id}_{noise_type}_train.npz")
        valid_file = os.path.join(compiled_dir, f"{record_id}_{noise_type}_valid.npz")
        
        if not os.path.exists(train_file) or not os.path.exists(valid_file):
            raise FileNotFoundError(f"Files for {noise_type} not found")
            
        train_data = np.load(train_file)
        valid_data = np.load(valid_file)
        
        model, history = train_model(
            train_data, 
            valid_data, 
            model_name=f"lstm_denoiser_{noise_type}",
            epochs=100  # More epochs for specific training
        )
    
    print("\nAll training complete!")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run Window.py first with the compile_datasets function.")
except Exception as e:
    print(f"Error during training: {str(e)}")






