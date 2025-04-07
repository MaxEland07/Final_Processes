import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

# Define directories
base_dir = "./MIT-BIH Arrhythmia Database"
compiled_dir = os.path.join(base_dir, "Compiled-Data")
results_dir = "./Results"
vis_dir = "./Visulisations/Filtering_Results"

# Create directories if they don't exist
for directory in [results_dir, vis_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Build the LSTM-based denoising model
def build_lstm_denoiser(input_shape=(512, 1)):
    """
    Builds the LSTM-based denoising model.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    print(model.summary())
    return model

# Training function
def train_model(train_data, valid_data, model_name="ecg_denoiser", epochs=50):
    """
    Train the LSTM denoising model.
    """
    # Load data
    X_train = train_data["noisy"]
    y_train = train_data["clean"]
    X_val = valid_data["noisy"] 
    y_val = valid_data["clean"]
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    
    # Compile the model
    model = build_lstm_denoiser(input_shape=(X_train.shape[1], X_train.shape[2]))
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
        batch_size=32,  # Larger batch size for faster training
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
    plt.title(f'ECG Denoiser Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, f"training_history.png"))
    plt.close()
    
    # Evaluate on validation set
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation MSE: {val_loss:.6f}")
    
    return model, history

# Main function
def main():
    try:
        # Check if the dataset files exist
        train_file = os.path.join(compiled_dir, "ecg_denoising_train.npz")
        test_file = os.path.join(compiled_dir, "ecg_denoising_test.npz")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError("Training/test dataset files not found")
        
        # Load datasets
        print(f"Loading datasets from {compiled_dir}...")
        train_data = np.load(train_file, allow_pickle=True)
        test_data = np.load(test_file, allow_pickle=True)
        
        print(f"Loaded training dataset with {train_data['noisy'].shape[0]} windows")
        print(f"Loaded test dataset with {test_data['noisy'].shape[0]} windows")
        
        # Configure memory usage for large datasets
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(f"GPU memory configuration error: {e}")
        
        # Train a model on all noise types and SNRs combined
        print("Training LSTM model on combined dataset...")
        
        # Use test data as validation
        model, history = train_model(
            train_data, 
            test_data, 
            model_name="ecg_denoiser",
            epochs=5
        )
        
        print("\nTraining complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run Window.py first to create the unified dataset.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()






