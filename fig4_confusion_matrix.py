import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix from paper (absolute counts)
cm = np.array([
    [17650, 112, 48, 10, 0],
    [61, 1880, 6, 3, 0],
    [44, 17, 4105, 24, 0],
    [8, 5, 9, 365, 3],
    [0, 0, 0, 5, 205]
])

# Convert to percentages for normalized view
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

classes = ['N', 'S', 'V', 'F', 'Q']
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Percentage (%)'})
plt.xlabel("Predicted class", fontsize=12)
plt.ylabel("True class", fontsize=12)
plt.title("Fig. 4: Confusion matrix (normalized, test set)", fontsize=12)
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=300)
plt.show()