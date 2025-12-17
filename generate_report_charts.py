"""
Script to generate charts and graphs for the project report
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load training history
with open('training_history.json', 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['accuracy']) + 1)

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

# 1. Training History - Accuracy and Loss
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Training Accuracy
axes[0, 0].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2.5)
axes[0, 0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2.5)
axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.7, 1.0])

# Training Loss
axes[0, 1].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2.5)
axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5)
axes[0, 1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(epochs, history['precision_3'], 'b-', label='Training Precision', linewidth=2.5)
axes[1, 0].plot(epochs, history['val_precision_3'], 'r-', label='Validation Precision', linewidth=2.5)
axes[1, 0].set_title('Model Precision Over Epochs', fontsize=14, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Precision', fontsize=12)
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0.7, 1.0])

# Recall
axes[1, 1].plot(epochs, history['recall_3'], 'b-', label='Training Recall', linewidth=2.5)
axes[1, 1].plot(epochs, history['val_recall_3'], 'r-', label='Validation Recall', linewidth=2.5)
axes[1, 1].set_title('Model Recall Over Epochs', fontsize=14, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Recall', fontsize=12)
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0.7, 1.0])

plt.tight_layout()
plt.savefig('report_training_history.png', dpi=300, bbox_inches='tight')
print("Saved: report_training_history.png")

# 2. Dataset Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training/Validation split
train_data = [6794, 1360]
labels = ['Training Set', 'Validation Set']
colors = ['#4CAF50', '#2196F3']
axes[0].pie(train_data, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0].set_title('Training vs Validation Data Distribution', fontsize=14, fontweight='bold', pad=15)

# Class distribution
fake_count = 2933 + 587  # train + validation fake
real_count = 3861 + 773   # train + validation real
class_data = [fake_count, real_count]
class_labels = ['Fake Banknotes', 'Real Banknotes']
class_colors = ['#F44336', '#4CAF50']
axes[1].bar(class_labels, class_data, color=class_colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel('Number of Images', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(class_data):
    axes[1].text(i, v + 50, str(v), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('report_dataset_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: report_dataset_distribution.png")

# 3. Performance Metrics Comparison
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
fake_metrics = [0.9660, 0.9743, 0.9660, 0.9701]
real_metrics = [0.9813, 0.9752, 0.9813, 0.9782]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, fake_metrics, width, label='Fake Banknotes', color='#F44336', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, real_metrics, width, label='Real Banknotes', color='#4CAF50', alpha=0.8, edgecolor='black')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics by Class', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0.9, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('report_performance_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: report_performance_metrics.png")

# 4. Confusion Matrix (from test results)
cm_data = [[682, 24], [19, 943]]  # From classification report: 706 fake, 962 real
cm = np.array(cm_data)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['FAKE', 'REAL'],
            yticklabels=['FAKE', 'REAL'],
            cbar_kws={'label': 'Count'},
            ax=ax,
            linewidths=2,
            linecolor='black')
ax.set_title('Confusion Matrix - Test Set Results', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

# Calculate percentages
total = cm.sum()
for i in range(2):
    for j in range(2):
        percentage = (cm[i, j] / cm[i].sum()) * 100
        ax.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('report_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: report_confusion_matrix.png")

# 5. Model Architecture Summary
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Model components
components = [
    'Input Layer\n(224×224×3)',
    'MobileNetV2\n(Pre-trained)',
    'Global Average\nPooling',
    'Batch Normalization\n+ Dropout (0.5)',
    'Dense Layer\n(128 units, ReLU)',
    'Batch Normalization\n+ Dropout (0.3)',
    'Dense Layer\n(64 units, ReLU)',
    'Dropout (0.2)',
    'Output Layer\n(1 unit, Sigmoid)'
]

y_positions = np.linspace(0.9, 0.1, len(components))
box_width = 0.15
box_height = 0.08

for i, (comp, y_pos) in enumerate(zip(components, y_positions)):
    # Draw box
    rect = plt.Rectangle((0.4 - box_width/2, y_pos - box_height/2), 
                        box_width, box_height,
                        facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.4, y_pos, comp, ha='center', va='center', 
            fontsize=9, fontweight='bold', wrap=True)
    
    # Draw arrow
    if i < len(components) - 1:
        ax.arrow(0.4, y_pos - box_height/2 - 0.01, 0, 
                -(y_positions[i] - y_positions[i+1] - box_height - 0.02),
                head_width=0.02, head_length=0.01, fc='black', ec='black')

ax.set_title('Model Architecture Overview', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('report_model_architecture.png', dpi=300, bbox_inches='tight')
print("Saved: report_model_architecture.png")

# 6. Learning Rate Schedule
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2.5, marker='o', markersize=4)
ax.set_title('Learning Rate Schedule During Training', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axvline(x=28, color='r', linestyle='--', linewidth=2, label='LR Reduction Point')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('report_learning_rate.png', dpi=300, bbox_inches='tight')
print("Saved: report_learning_rate.png")

print("\nAll charts generated successfully!")

