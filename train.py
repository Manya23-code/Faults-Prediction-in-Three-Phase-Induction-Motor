import pandas as pd
import numpy as np
import os
from model_arch import get_flux_only_model

def train_agent():
    filename = 'data/motor_data_temp.csv' # Use your custom CSV here later
    
    if not os.path.exists(filename):
        print(f"❌ ERROR: Could not find {filename}")
        return

    # 1. LOAD DATA
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    
    # 2. PREPROCESS (Flux Only)
    features = df[['X-axis', 'Y-axis', 'Z-axis']].values
    X_flux, Y_load = [], []
    
    # Creating windows of 1000 samples
    for i in range(0, len(features) - 1000, 1000):
        X_flux.append(features[i:i+1000])
        Y_load.append(df['Load'].iloc[i]) # This label depends on your specific data file
        
    X_flux = np.array(X_flux)
    Y_load = np.array(Y_load).reshape(-1, 1)

    # 3. TRAIN
    model = get_flux_only_model(window_size=1000)
    print("🚀 Training Magnetic-Only AI Agent...")
    model.fit(X_flux, Y_load, epochs=10, verbose=1)
    
    # 4. SAVE
    model.save('motor_brain.h5')
    print("✅ SUCCESS: Flux-only 'motor_brain.h5' created!")

if __name__ == "__main__":
    train_agent()