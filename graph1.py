# =======================================================
# STEP 6: PPT VISUALIZATIONS (GRAPHS)
# =======================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("\n7. Generating PPT Graphs...")

# GRAPH 1: 5-Fold Cross Validation Bar Chart
plt.figure(figsize=(10, 5))
fold_labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']

bars = plt.bar(fold_labels, cv_scores, color=colors)
plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label=f'Average: {np.mean(cv_scores):.2f}%')

# Add exact numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 110)
plt.title('SMOTE + CNN-LSTM: 5-Fold Cross Validation Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig('kfold_results.png', dpi=300) # Saves the graph as a high-quality image
plt.show()

# GRAPH 2: The Confusion Matrix (Using the data from the last fold)
# Make predictions using the trained model
y_pred = model.predict(X_val)

# Convert predictions and actual labels from One-Hot to normal numbers (0, 1, 2, 3, 4)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Generate Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
class_names = ['No Load', 'Normal Load', 'High Res.', 'Overload', 'Phase Open']

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, 
            annot_kws={"size": 14, "weight": "bold"})
plt.title('AI Brain: Confusion Matrix (Fold 5 Test Data)', fontsize=16, fontweight='bold')
plt.ylabel('Actual Motor State (True Truth)', fontsize=12)
plt.xlabel('AI Predicted State', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300) # Saves the graph as a high-quality image
plt.show()

print("Graphs successfully generated and saved as PNG images!")