import pandas as pd
import numpy as np
from scipy.fft import fft
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical

print("1. Loading the PERFECTLY CLEANED Dataset...")
# Ab hum seedha 1.py ka banaya hua clean data use kar rahe hain!
file_path = r"C:\SEC project\hybrid_motor_data.csv"
df = pd.read_csv(file_path)

# Safely extract columns
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['amplified flux'] = pd.to_numeric(df['amplified flux'], errors='coerce')
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
df = df.dropna(subset=['time', 'amplified flux', 'speed'])

print(f"✅ Data Ready! Total Valid Rows: {len(df)}")

print("2. Normalizing the Features (Scaling Speed & Flux)...")
df['amplified flux'] = (df['amplified flux'] - df['amplified flux'].min()) / (df['amplified flux'].max() - df['amplified flux'].min())
df['speed'] = (df['speed'] - df['speed'].min()) / (df['speed'].max() - df['speed'].min())

print("3. Assigning ALL 5 Classes based on Lab Notebook Timestamps...")
df['Label'] = 1 # Normal (Default)
df.loc[(df['time'] >= 190.0) & (df['time'] <= 257.99), 'Label'] = 0  # No Load
df.loc[(df['time'] >= 258.0) & (df['time'] <= 350.00), 'Label'] = 1  # Normal
df.loc[(df['time'] >= 468.0) & (df['time'] <= 492.99), 'Label'] = 2  # High Res
df.loc[(df['time'] >= 506.0) & (df['time'] <= 515.99), 'Label'] = 2  # High Res
df.loc[(df['time'] >= 353.0) & (df['time'] <= 467.99), 'Label'] = 3  # Overload
df.loc[(df['time'] >= 493.0) & (df['time'] <= 505.99), 'Label'] = 4  # Phase Open
df.loc[(df['time'] >= 516.0) & (df['time'] <= 524.99), 'Label'] = 4  # Phase Open

print("4. Time-Windows and FFT (Combining Flux & Speed)...")
X_raw, y_true = [], []
for i in range(0, len(df) - 15, 2): 
    # Feeding BOTH features to the AI
    X_raw.append(df[['amplified flux', 'speed']].iloc[i : i + 15].values)
    y_true.append(df['Label'].iloc[i + 15 - 1]) 

X_fft = np.abs(fft(np.array(X_raw), axis=1))
X_flat = X_fft.reshape(X_fft.shape[0], -1)

print("5. Applying SMOTE for 5 Classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, np.array(y_true))
X_final = X_resampled.reshape(-1, 15, 2) # Shape: 15 time steps, 2 features (Flux & Speed)
y_final = to_categorical(y_resampled, num_classes=5) 

print("6. Building & Training the Dual-Feature Brain...")
inputs = Input(shape=(15, 2))
x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = LSTM(64, return_sequences=True)(x)
attn_out = Attention()([x, x])
x = GlobalAveragePooling1D()(attn_out)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(5, activation='softmax')(x) 

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the Model
model.fit(X_final, y_final, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

print("7. Saving the Upgraded Brain...")
model.save(r"C:\SEC project\best_attention_motor_model.h5")
print("\n🎉 MODEL SAVED! The AI has successfully learned from both Flux and Speed!")