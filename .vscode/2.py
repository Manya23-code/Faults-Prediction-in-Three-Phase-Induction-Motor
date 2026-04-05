
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
# Dhyan dijiye: Is niche wali line ke aakhir mein 'Input' add ho gaya hai
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

# =========================================================
# STEP 1: LOAD THE PERFECTED DATASET
# =========================================================
print("1. Loading the 5-Column Hybrid Dataset...")
df = pd.read_csv('hybrid_motor_data.csv')
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

print("Data loaded successfully as purely numeric!")

# =========================================================
# STEP 2: THE 5-CLASS ADVANCED LABELING
# =========================================================
print("2. Assigning 5-State Operating Labels...")

# Default state: Assume '1' (Normal Load / Healthy) for most of the data
df['Label'] = 1 

# ---> CHANGEME: Yahan apne 'No Load' wale seconds daalein <---
# (Example: 0 to 150 seconds)
df.loc[(df['Timestamp'] >= 190.018) & (df['Timestamp'] <= 257.996), 'Label'] = 0  

# ---> CHANGEME: Yahan 'High Resistance Normal' wale seconds daalein <---
# (Example: 200 to 280 seconds)
df.loc[(df['Timestamp'] >= 468.080) & (df['Timestamp'] <= 492.926), 'Label'] = 2
df.loc[(df['Timestamp'] >= 506.046) & (df['Timestamp'] <= 515.968), 'Label'] = 2  

# FAULTS (Based on your exact notes)
df.loc[(df['Timestamp'] >= 353.034) & (df['Timestamp'] <= 467.998), 'Label'] = 3  # Overload Fault
df.loc[(df['Timestamp'] >= 493.008) & (df['Timestamp'] <= 505.964), 'Label'] = 4  # Phase Open Fault
df.loc[(df['Timestamp'] >= 516.050) & (df['Timestamp'] <= 524.414), 'Label'] = 4  # Phase Open Fault

# =========================================================
# STEP 3: HYBRID PHYSICS FEATURE ENGINEERING
# =========================================================
print("3. Calculating Physics-Informed Internal Flux...")
f = 50.0        # Frequency in Hz
N = 250.0       # Assumed Number of turns
Kw = 0.955      # Standard distributed winding factor
Z = 2.5         # Assumed Stator impedance (Ohms)

# Calculate Theoretical Flux using V and I
df['EMF'] = df['Voltage'] - (df['Current'] * Z)
df['Calculated_Flux'] = df['EMF'] / (4.44 * f * N * Kw)

# Normalization (Scaling both between 0 and 1 so AI doesn't get confused)
df['Amplified_Flux'] = (df['Amplified_Flux'] - df['Amplified_Flux'].min()) / (df['Amplified_Flux'].max() - df['Amplified_Flux'].min())
df['Calculated_Flux'] = (df['Calculated_Flux'] - df['Calculated_Flux'].min()) / (df['Calculated_Flux'].max() - df['Calculated_Flux'].min())

# =========================================================
# STEP 4: DATA WINDOWING (Fixed for 12Hz Sampling Rate)
# =========================================================
print("4. Slicing data into overlapping AI Windows...")
def create_hybrid_windows(data, labels, window_size=15, step=2):
    X, y = [], []
    # Step size 2 hone se AI har 0.16 seconds mein ek nayi photo lega!
    for i in range(0, len(data) - window_size, step): 
        X.append(data.iloc[i : i + window_size].values)
        y.append(labels.iloc[i + window_size - 1]) 
    return np.array(X), np.array(y)

# Use window_size=15 (approx 1.2 seconds per window)
X, y = create_hybrid_windows(df[['Amplified_Flux', 'Calculated_Flux']], df['Label'], window_size=15, step=2)

X = X.reshape(X.shape[0], X.shape[1], 2) 
y = to_categorical(y, num_classes=5)
print("Shuffling and splitting data like a pro...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check total windows generated
print(f"MAGIC: Generated {X.shape[0]} training examples from just 4048 rows!")

# =========================================================
# STEP 5: BALANCING THE AI (Fixed Class Weights!)
# =========================================================
# Ab hum Faults ko 10 guna nahi, sirf 2 guna (2x) importance denge.
weight_for_classes = {
    0: 1.0,   # No Load
    1: 1.0,   # Normal Load
    2: 1.5,   # High Resistance (Thoda extra dhyan)
    3: 2.0,   # OVERLOAD (Sirf 2x bias)
    4: 2.5    # PHASE OPEN (Rare hai, toh 2.5x bias)
}



# =========================================================
# STEP 6: BUILDING THE HYBRID 1D-CNN ARCHITECTURE (Shape Fixed!)
# =========================================================
print("5. Building the Hybrid 1D-CNN Brain...")
model = Sequential([
    # Naya aur safe tareeka: Keras Input Layer (Yeh mismatch error ko rokega)
    Input(shape=(15, 2)), 
    
    # Ab Conv1D mein input_shape likhne ki zaroorat nahi hai
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=16, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(5, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =========================================================
# STEP 7: TRAINING & SAVING
# =========================================================
print("6. Training the Master AI... (Watch the accuracy rise!)")

# Yahan humne X ki jagah X_train aur validation_split ki jagah validation_data laga diya hai
history = model.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=64, 
    validation_data=(X_test, y_test), 
    class_weight=weight_for_classes
)

model.save('final_sec_motor_model.h5')
print("\n PROJECT COMPLETE: Master AI Trained and Saved as 'final_sec_motor_model.h5'!")