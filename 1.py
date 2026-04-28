import pandas as pd
import numpy as np

# Apne paths check kar lijiye
input_file = r"C:\SEC project\.vscode\INDUCTION.txt"
output_file = r"C:\SEC project\hybrid_motor_data.csv"

print("1. Reading raw text file safely (Line-by-Line)...")
valid_data = []

# Open the file and read it manually to bypass Pandas parsing errors
with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # line.strip().split() removes extra spaces and splits purely by ANY whitespace/tab
        parts = line.strip().split()
        
        # Keep ONLY lines that have at least 5 elements
        if len(parts) >= 5:
            # Take exactly the first 5 elements
            valid_data.append(parts[:5])

print(f"Successfully extracted {len(valid_data)} clean rows.")

print("2. Converting to strictly numeric DataFrame...")
# Build DataFrame directly with 5 columns
df = pd.DataFrame(valid_data, columns=['Timestamp', 'Col2', 'Flux', 'Voltage', 'Current'])

# Safely convert strings to actual numbers
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows that became NaN during conversion
initial_len = len(df)
df = df.dropna()
print(f"Dropped {initial_len - len(df)} corrupted rows (if any).")

print("3. Saving the fixed dataset...")
df.to_csv(output_file, index=False)

# ==========================================
# THE DIAGNOSTIC CHECK (Truth Teller)
# ==========================================
overload_count = len(df[(df['Timestamp'] >= 353.0) & (df['Timestamp'] <= 467.9)])
phase_open_count = len(df[(df['Timestamp'] >= 493.0) & (df['Timestamp'] <= 524.4)])

print("\n=== DIAGNOSTIC REPORT ===")
print(f"Total Rows Saved: {len(df)}")
print(f"Overload Rows Recovered: {overload_count}")
print(f"Phase Open Rows Recovered: {phase_open_count}")

if overload_count > 0:
    print("\nSUCCESS! Your missing fault data is officially back! 🎉")
    print("Next Steps: Run 2.py to train the AI on this new data, then run 3_graphs.py!")
else:
    print("\nWARNING: The data is still not found. Check if INDUCTION.txt actually contains the 350s-470s range.")