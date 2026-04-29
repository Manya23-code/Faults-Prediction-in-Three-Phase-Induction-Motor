import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from scipy.fft import fft

print("1. Loading the New Trained AI Brain and Data...")
model = load_model(r"C:\SEC project\best_attention_motor_model.h5")
df = pd.read_csv(r"C:\SEC project\hybrid_motor_data.csv")

# Ensure proper formatting
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['amplified flux'] = pd.to_numeric(df['amplified flux'], errors='coerce')
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
df = df.dropna(subset=['time', 'amplified flux', 'speed'])

print("2. Normalizing and Labeling (Same as Training)...")
df['amplified flux'] = (df['amplified flux'] - df['amplified flux'].min()) / (df['amplified flux'].max() - df['amplified flux'].min())
df['speed'] = (df['speed'] - df['speed'].min()) / (df['speed'].max() - df['speed'].min())

df['Label'] = 1 # Normal (Default)
df.loc[(df['time'] >= 190.0) & (df['time'] <= 257.99), 'Label'] = 0
df.loc[(df['time'] >= 258.0) & (df['time'] <= 350.00), 'Label'] = 1
df.loc[(df['time'] >= 468.0) & (df['time'] <= 492.99), 'Label'] = 2
df.loc[(df['time'] >= 506.0) & (df['time'] <= 515.99), 'Label'] = 2
df.loc[(df['time'] >= 353.0) & (df['time'] <= 467.99), 'Label'] = 3
df.loc[(df['time'] >= 493.0) & (df['time'] <= 505.99), 'Label'] = 4
df.loc[(df['time'] >= 516.0) & (df['time'] <= 524.99), 'Label'] = 4

print("3. Processing Data (FFT)...")
X_raw, y_true = [], []
for i in range(0, len(df) - 15, 2): 
    X_raw.append(df[['amplified flux', 'speed']].iloc[i : i + 15].values)
    y_true.append(df['Label'].iloc[i + 15 - 1]) 

X_fft = np.abs(fft(np.array(X_raw), axis=1))

print("4. AI is taking the Test...")
y_pred_probs = model.predict(X_fft)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = ['No Load', 'Normal Load', 'High Res', 'Overload', 'Phase Open']

print("5. Generating the Final 5x5 Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            annot_kws={"size": 14, "weight": "bold"})
plt.title('CNN-LSTM Model: Flux + Speed', fontsize=16, fontweight='bold')
plt.ylabel('Actual Hardware State', fontsize=12, fontweight='bold')
plt.xlabel('AI Predicted State', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('Final_Confusion_Matrix.png', dpi=300)
plt.show()

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))