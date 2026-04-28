import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from scipy.fft import fft

print("1. Loading the Trained AI Brain and Hardware Data...")
model = load_model(r"C:\SEC project\best_attention_motor_model.h5")
df = pd.read_csv(r"C:\SEC project\hybrid_motor_data.csv")

# Ensure proper formatting
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df['Flux'] = pd.to_numeric(df['Flux'], errors='coerce')
df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
df = df.dropna(subset=['Timestamp', 'Flux', 'Voltage'])

print("2. Assigning ALL 5 Classes based on Lab Notebook...")
df['Label'] = 1 # Normal (Default)
df.loc[(df['Timestamp'] >= 190.0) & (df['Timestamp'] <= 257.99), 'Label'] = 0  # No Load
df.loc[(df['Timestamp'] >= 258.0) & (df['Timestamp'] <= 350.00), 'Label'] = 1  # Normal
df.loc[(df['Timestamp'] >= 468.0) & (df['Timestamp'] <= 492.99), 'Label'] = 2  # High Res
df.loc[(df['Timestamp'] >= 506.0) & (df['Timestamp'] <= 515.99), 'Label'] = 2  # High Res
df.loc[(df['Timestamp'] >= 353.0) & (df['Timestamp'] <= 467.99), 'Label'] = 3  # Overload
df.loc[(df['Timestamp'] >= 493.0) & (df['Timestamp'] <= 505.99), 'Label'] = 4  # Phase Open
df.loc[(df['Timestamp'] >= 516.0) & (df['Timestamp'] <= 524.99), 'Label'] = 4  # Phase Open

print("3. Processing Data (FFT)...")
f, N, Kw = 50.0, 250.0, 0.955
df['Calculated_Flux'] = df['Voltage'] / (4.44 * f * N * Kw) 

df['Flux'] = (df['Flux'] - df['Flux'].min()) / (df['Flux'].max() - df['Flux'].min())
df['Calculated_Flux'] = (df['Calculated_Flux'] - df['Calculated_Flux'].min()) / (df['Calculated_Flux'].max() - df['Calculated_Flux'].min())

X_raw, y_true = [], []
for i in range(0, len(df) - 15, 2): 
    X_raw.append(df[['Flux', 'Calculated_Flux']].iloc[i : i + 15].values)
    y_true.append(df['Label'].iloc[i + 15 - 1]) 

X_fft = np.abs(fft(np.array(X_raw), axis=1))

print("4. AI is making predictions...")
y_pred_probs = model.predict(X_fft)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = ['No Load', 'Normal Load', 'High Res', 'Overload', 'Phase Open']

print("5. Generating the Perfect 5x5 Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            annot_kws={"size": 14, "weight": "bold"})
plt.title('CNN-LSTM Model: Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Hardware State', fontsize=12, fontweight='bold')
plt.xlabel('AI Predicted State', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('ML_Confusion_Matrix.png', dpi=300)
plt.show()

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))