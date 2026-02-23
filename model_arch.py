import tensorflow as tf
from tensorflow.keras import layers, models

def get_flux_only_model(window_size=1000):
    # Input: (1000 samples, 3 axes: X, Y, Z)
    flux_input = layers.Input(shape=(window_size, 3), name="flux_input")
    
    # Feature Extraction (Convolutional Layers)
    x = layers.Conv1D(32, 3, activation='relu')(flux_input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x) 

    # Decision Layers (Dense)
    z = layers.Dense(64, activation='relu')(x)
    z = layers.Dense(32, activation='relu')(z)
    
    # Final Output: Predicting the Load (Percentage or Watts)
    output = layers.Dense(1, activation='linear', name="load_output")(z)

    model = models.Model(inputs=flux_input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model