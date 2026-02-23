import tensorflow as tf
import numpy as np
import pandas as pd
import os

# 1. Define the model path
model_path = 'motor_brain.h5'
data_path = 'data/motor_data_temp.csv'

# 2. Check if the model exists before loading
if not os.path.exists(model_path):
    print(f"❌ ERROR: {model_path} not found. Run 'python train.py' first!")
else:
    # --- LOAD THE MODEL ---
    # This defines the 'agent' variable that was missing
    agent = tf.keras.models.load_model(model_path)
    print(" AI Agent loaded successfully.")

    # 3. Prepare a sample of data for testing
    if os.path.exists(data_path):
        # Load real data from your temp file
        df = pd.read_csv(data_path)
        # Take the first 1000 samples (the window size)
        sample_flux = df[['X-axis', 'Y-axis', 'Z-axis']].values[:1000]
        # Reshape to (1, 1000, 3) because the AI expects a "batch" of windows
        flux_in = np.expand_dims(sample_flux, axis=0)
        
        # --- MAKE THE PREDICTION ---
        prediction = agent.predict(flux_in, verbose=0)
        
        print(f"\n AI Prediction Result: {prediction[0][0]:.2f}% Load")
    else:
        print(f" Warning: {data_path} not found. Using random dummy data for test.")
        # Fallback to random data if the CSV is missing
        flux_in = np.random.random((1, 1000, 3))
        prediction = agent.predict(flux_in, verbose=0)
        print(f" Dummy Prediction Result: {prediction[0][0]:.2f}")