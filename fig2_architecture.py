import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Layer positions (x, y, width, height)
layers = [
    ("Input\n(300,1)", 1, 4.5, 1.2, 0.6),
    ("Conv1D\n32, k=5", 2.6, 4.5, 1.2, 0.6),
    ("MaxPool\n(2)", 4.2, 4.5, 1.0, 0.6),
    ("Conv1D\n64, k=5", 5.6, 4.5, 1.2, 0.6),
    ("MaxPool\n(2)", 7.2, 4.5, 1.0, 0.6),
    ("Conv1D\n128, k=5", 8.6, 4.5, 1.2, 0.6),
    ("MaxPool\n(2)", 10.2, 4.5, 1.0, 0.6),
    ("LSTM\n64 units", 2.5, 2.8, 1.3, 0.6),
    ("GlobalAvg\nPooling", 4.3, 2.8, 1.2, 0.6),
    ("Dropout\n(0.3)", 6.0, 2.8, 1.0, 0.6),
    ("Dense\n(5)", 7.5, 2.8, 1.0, 0.6),
    ("Softmax\nOutput", 9.0, 2.8, 1.0, 0.6),
]

colors = ['lavender', 'lightgreen', 'lightcoral', 'lightgreen', 'lightcoral',
          'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow', 'lightgray',
          'lightblue', 'lightblue']

for (label, x, y, w, h), color in zip(layers, colors):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=8, fontweight='bold')

# Arrows between layers
arrows = [(1.6, 4.8, 2.6, 4.8), (3.8, 4.8, 4.2, 4.8), (5.2, 4.8, 5.6, 4.8),
          (6.8, 4.8, 7.2, 4.8), (8.2, 4.8, 8.6, 4.8), (9.8, 4.8, 10.2, 4.8),
          (2.6, 4.5, 2.6, 3.4), (3.8, 3.1, 4.3, 3.1), (5.5, 3.1, 6.0, 3.1),
          (7.0, 3.1, 7.5, 3.1), (8.5, 3.1, 9.0, 3.1)]

for (x1, y1, x2, y2) in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'))

plt.title("Fig. 2: Hybrid CNN-LSTM architecture for ECG classification", fontsize=11)
plt.tight_layout()
plt.savefig("fig2_cnn_lstm_architecture.png", dpi=300, bbox_inches='tight')
plt.show()