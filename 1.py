import pandas as pd
import os

print("1. Reading raw text file safely (sec.txt)...")
input_file = r"C:\SEC project\sec.txt"
output_file = r"C:\SEC project\hybrid_motor_data.csv"

try:
    # Read the data, ignoring any bad or incomplete lines
    df = pd.read_csv(input_file, encoding='utf-8-sig', on_bad_lines='skip')

    # Clean up column names (remove extra spaces and make lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert everything to numbers, replacing garbage values with blank (NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop any rows that have missing data
    df = df.dropna()

    print(f"Successfully extracted {len(df)} clean rows.")
    print("2. Converting to strictly numeric DataFrame...")

    # Save to the final CSV that 2.py uses
    df.to_csv(output_file, index=False)
    
    print("3. Saving the fixed dataset...\n")
    print("=== DIAGNOSTIC REPORT ===")
    print(f"Total Rows Saved: {len(df)}")
    print(f"Columns Found: {df.columns.tolist()}")
    print("✅ SUCCESS: hybrid_motor_data.csv is perfectly generated and ready for training!")

except FileNotFoundError:
    print("❌ ERROR: sec.txt file not found! Please check if the name is exactly sec.txt in your folder.")
except Exception as e:
    print(f"❌ ERROR: Something went wrong -> {e}")