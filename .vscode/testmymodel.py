import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import random

print("1. Waking up the AI Brain...")
model = load_model('final_sec_motor_model.h5')

print("2. Loading Motor Data and Assigning Labels...")
df = pd.read_csv('hybrid_motor_data.csv')

# ==========================================
# THE MISSING PIECE: Re-creating the Labels
# ==========================================
df['Label'] = 1 
df.loc[(df['Timestamp'] >= 0) & (df['Timestamp'] <= 150), 'Label'] = 0  
df.loc[(df['Timestamp'] >= 200) & (df['Timestamp'] <= 280), 'Label'] = 2  
df.loc[(df['Timestamp'] >= 353) & (df['Timestamp'] <= 467), 'Label'] = 3  
df.loc[(df['Timestamp'] >= 493) & (df['Timestamp'] <= 506), 'Label'] = 4  
df.loc[(df['Timestamp'] >= 516) & (df['Timestamp'] <= 524), 'Label'] = 4  

# ==========================================
# FEATURE ENGINEERING
# ==========================================
Z = 2.5
df['EMF'] = df['Voltage'] - (df['Current'] * Z)
df['Calculated_Flux'] = df['EMF'] / (4.44 * 50.0 * 250.0 * 0.955)

df['Amplified_Flux'] = (df['Amplified_Flux'] - df['Amplified_Flux'].min()) / (df['Amplified_Flux'].max() - df['Amplified_Flux'].min())
df['Calculated_Flux'] = (df['Calculated_Flux'] - df['Calculated_Flux'].min()) / (df['Calculated_Flux'].max() - df['Calculated_Flux'].min())

states = {
    0: " NO LOAD",
    1: " NORMAL LOAD",
    2: " HIGH RESISTANCE",
    3: " OVERLOAD FAULT!",
    4: " PHASE OPEN FAULT!"
}

def true_random_test():
    # 1. Pick a completely random starting point
    start = random.randint(0, len(df) - 20)
    
    # 2. Extract the 15-row window for AI
    test_window = df[['Amplified_Flux', 'Calculated_Flux']].iloc[start : start + 15].values
    test_window = test_window.reshape(1, 15, 2)
    
    # 3. Extract the ACTUAL Label from our recreated Label column
    actual_label_id = df['Label'].iloc[start : start + 15].mode()[0]
    
    # 4. Ask AI to Predict
    prediction = model.predict(test_window, verbose=0)
    predicted_label_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # 5. Compare and Print!
    print("\n" + "="*55)
    print(f" RANDOM TEST: Analyzing Rows {start} to {start+15}")
    print("-" * 55)
    
    print(f" ACTUAL STATE (From Data) : {states[actual_label_id]}")
    
    # Check if AI is correct
    if predicted_label_id == actual_label_id:
        print(f" AI PREDICTION (Matches!)  : {states[predicted_label_id]} yes! ")
    else:
        print(f" AI PREDICTION (Wrong)     : {states[predicted_label_id]} no! ")
        
    print(f" Confidence Score         : {confidence:.2f}%")
    print("="*55)

# Run 5 completely random tests!
print("\n RUNNING 5 BLIND RANDOM TESTS...\n")
for _ in range(5):
    true_random_test()

print("\n" + "="*55)
print(" TARGETED TEST: FORCING AI TO LOOK AT SINGLE PHASE FAULT")
print("="*55)

# Row 3650 is exactly around 495 seconds (Middle of Phase Open Fault)
start = 3650
test_window = df[['Amplified_Flux', 'Calculated_Flux']].iloc[start : start + 15].values
test_window = test_window.reshape(1, 15, 2)

prediction = model.predict(test_window, verbose=0)
predicted_label_id = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f" AI PREDICTION  : {states[predicted_label_id]}")
print(f" Confidence     : {confidence:.2f}%")
print("="*55 + "\n")    