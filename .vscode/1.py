from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
import pandas as pd

# File paths (Output file ka naam hybrid_motor_data rakha hai taaki AI training file se match kare)
input_file = r'C:\SEC project\.vscode\FINAL 1.txt'
output_file = r'C:\SEC project\hybrid_motor_data.csv'

# Step 1: Nayi CSV banayein poore 5 Headers ke saath
columns_list = ['Timestamp', 'Raw_Flux', 'Amplified_Flux', 'Voltage', 'Current']
headers = pd.DataFrame(columns=columns_list)
headers.to_csv(output_file, index=False)

print("Starting safe 5-column conversion...")

# Step 2: Chunking (Memory Safe Read)
chunk_size = 10000
chunks = pd.read_csv(input_file, sep=',', chunksize=chunk_size, names=columns_list, on_bad_lines='skip')

total_rows = 0

# Step 3: Clean and save each chunk
for chunk in chunks:
    # Jo rows pehle se hi khali/tooti hui hain, unhe hatao
    chunk = chunk.dropna()
    
    # SABHI 5 columns ko strictly Numbers mein badlo. 
    # Agar kisi mein text ("no load", "V", "A") aaya, toh use 'NaN' (blank) bana do.
    for col in columns_list:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    
    # Ab unn saari rows ko permanently delete kar do jisme text ki wajah se 'NaN' ban gaya tha
    chunk = chunk.dropna()
    
    # Saaf data ko CSV mein append (jod) do
    chunk.to_csv(output_file, mode='a', header=False, index=False)
    
    total_rows += len(chunk)
    print(f"Processed {total_rows} rows safely...")

print("Conversion Complete! Your clean 5-column CSV is ready for AI Training.")