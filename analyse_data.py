import pandas as pd
import numpy as np

print("=== DATASET HEALTH CHECK (DETECTIVE MODE) ===")

# 1. Load the Data
df = pd.read_csv('hybrid_motor_data.csv')
print(f"Total Rows in CSV: {len(df)}")

# 2. Clean and verify column types
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df['Flux'] = pd.to_numeric(df['Flux'], errors='coerce')
df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
df = df.dropna(subset=['Timestamp', 'Flux', 'Voltage'])
print(f"Valid Rows after cleaning: {len(df)}\n")

# 3. Apply the exact labelling logic we are feeding to the AI
df['Label'] = 1 # Normal Load (Default)
df.loc[(df['Timestamp'] >= 190.0) & (df['Timestamp'] <= 257.9), 'Label'] = 0  # No Load
df.loc[(df['Timestamp'] >= 468.0) & (df['Timestamp'] <= 492.9), 'Label'] = 2  # High Res
df.loc[(df['Timestamp'] >= 506.0) & (df['Timestamp'] <= 515.9), 'Label'] = 2  # High Res
df.loc[(df['Timestamp'] >= 353.0) & (df['Timestamp'] <= 467.9), 'Label'] = 3  # Overload
df.loc[(df['Timestamp'] >= 493.0) & (df['Timestamp'] <= 505.9), 'Label'] = 4  # Phase Open
df.loc[(df['Timestamp'] >= 516.0) & (df['Timestamp'] <= 524.4), 'Label'] = 4  # Phase Open

class_names = {0: 'No Load', 1: 'Normal Load', 2: 'High Res', 3: 'Overload', 4: 'Phase Open'}

# --- TEST 1: CLASS IMBALANCE (Did we actually label them right?) ---
print("--- TEST 1: ROW COUNT PER FAULT (Are any labels missing?) ---")
class_counts = df['Label'].value_counts().sort_index()
for idx, count in class_counts.items():
    print(f"[{idx}] {class_names[idx].ljust(15)}: {count} rows")

# --- TEST 2: THE PHYSICS CHECK (Are the sensor values actually different?) ---
print("\n--- TEST 2: SENSOR PHYSICS CHECK (Average Flux & Voltage per state) ---")
physics = df.groupby('Label')[['Flux', 'Voltage']].mean()
for idx, row in physics.iterrows():
    print(f"{class_names[idx].ljust(15)}: Avg Flux = {row['Flux']:>6.2f} | Avg Voltage = {row['Voltage']:>6.2f}")

# --- TEST 3: THE VARIANCE CHECK (Is the data dead/frozen?) ---
print("\n--- TEST 3: SENSOR VIBRATION/NOISE (Standard Deviation) ---")
variance = df.groupby('Label')['Flux'].std()
for idx, std in variance.items():
    print(f"{class_names[idx].ljust(15)}: Flux Variation (Noise) = {std:>6.2f}")