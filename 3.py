import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from scipy.fft import fft

print("1. Loading the Trained AI Brain...")
# Load the saved model that 2.py created
model = load_model(r"C:\SEC project\best_attention_motor_model.h5")

# =======================================================
# GRAPH 1: THE K-FOLD BAR CHART
# =======================================================
print("2. Generating K-Fold Results Graph...")
# Your actual scores from the successful training
cv_scores = [100.00, 83.47, 66.12, 86.36, 96.27]
fold_labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']

plt.figure(figsize=(10, 5))
bars = plt.bar(fold_labels, cv_scores, color=colors)
plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label=f'Average: {np.mean(cv_scores):.2f}%')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 110)
plt.title('SMOTE + CNN-LSTM: 5-Fold Cross Validation Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig('kfold_results.png', dpi=300) 
plt.close() # Closes the background plot to save memory

# =======================================================
# GRAPH 2: THE CONFUSION MATRIX
# =======================================================
print("3. Preparing Data for Confusion Matrix...")
# Load and prep data quickly just for prediction
df = pd.read_csv(r"C:\SEC project\hybrid_motor_data.csv")
df.columns = ['Timestamp', 'Col2', 'Flux', 'Voltage', 'Current']

df['Label'] = 1 
df.loc[(df['Timestamp'] >= 190.018) & (df['Timestamp'] <= 257.996), 'Label'] = 0  
df.loc[(df['Timestamp'] >= 468.080) & (df['Timestamp'] <= 492.926), 'Label'] = 2  
df.loc[(df['Timestamp'] >= 506.046) & (df['Timestamp'] <= 515.968), 'Label'] = 2  
df.loc[(df['Timestamp'] >= 353.034) & (df['Timestamp'] <= 467.998), 'Label'] = 3  
df.loc[(df['Timestamp'] >= 493.008) & (df['Timestamp'] <= 505.964), 'Label'] = 4  
df.loc[(df['Timestamp'] >= 516.050) & (df['Timestamp'] <= 524.414), 'Label'] = 4  

f, N, Kw = 50.0, 250.0, 0.955
df['Calculated_Flux'] = df['Voltage'] / (4.44 * f * N * Kw) 
df['Flux'] = (df['Flux'] - df['Flux'].min()) / (df['Flux'].max() - df['Flux'].min())
df['Calculated_Flux'] = (df['Calculated_Flux'] - df['Calculated_Flux'].min()) / (df['Calculated_Flux'].max() - df['Calculated_Flux'].min())

X_raw, y_true = [], []
for i in range(0, len(df) - 15, 2): 
    X_raw.append(df[['Flux', 'Calculated_Flux']].iloc[i : i + 15].values)
    y_true.append(df['Label'].iloc[i + 15 - 1]) 

# Apply FFT and predict
X_fft = np.abs(fft(np.array(X_raw), axis=1))
print("4. AI is analyzing the data...")
y_pred_probs = model.predict(X_fft)
y_pred = np.argmax(y_pred_probs, axis=1)

print("5. Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
class_names = ['No Load', 'Normal Load', 'High Resistance', 'Overload', 'Phase Open']

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, 
            annot_kws={"size": 14, "weight": "bold"})
plt.title('AI Brain: Total Dataset Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Motor State (True Truth)', fontsize=12)
plt.xlabel('AI Predicted State', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

print("\nSUCCESS! Both graphs have been saved in your project folder.")