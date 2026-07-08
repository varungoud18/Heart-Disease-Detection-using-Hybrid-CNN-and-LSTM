import os
import numpy as np
import matplotlib.pyplot as plt

# Get project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

models = ['CNN-only', 'LSTM-only', 'Yildirim (2020)', 'Ours (CNN-LSTM)']
accuracy = [94.2, 91.8, 99.0, 98.7]
macro_f1 = [85.6, 83.2, 97.1, 97.4]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='skyblue')
rects2 = ax.bar(x + width/2, macro_f1, width, label='Macro F1 (%)', color='salmon')

ax.set_ylabel('Percentage (%)')
ax.set_title('Fig. 5: Performance comparison across models (patient-wise split)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()

plot_path = os.path.join(ROOT_DIR, 'overleaf_images', 'fig5_performance_bars.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
plt.show()
