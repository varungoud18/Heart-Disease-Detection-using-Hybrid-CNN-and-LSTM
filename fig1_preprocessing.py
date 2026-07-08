import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.axis('off')

# Define boxes (x0, y0, width, height)
steps = [
    ("Raw ECG\nSignal", 0.5, 0.35, 1.2, 0.3),
    ("High-pass filter\n(0.5 Hz)", 2.2, 0.35, 1.2, 0.3),
    ("Z-score\nNormalization", 3.9, 0.35, 1.2, 0.3),
    ("Pan-Tompkins\nR-peak detection", 5.6, 0.35, 1.5, 0.3),
    ("Slice 300 samples\n(R_i ± 149)", 7.6, 0.35, 1.5, 0.3),
]

for i, (label, x, y, w, h) in enumerate(steps):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                  facecolor='lightblue', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows
for i in range(len(steps)-1):
    start_x = steps[i][1] + steps[i][3]
    end_x = steps[i+1][1]
    ax.annotate('', xy=(end_x, 0.5), xytext=(start_x, 0.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

plt.title("Fig. 1: Preprocessing pipeline for heartbeat window extraction", fontsize=10)
plt.tight_layout()
plt.savefig("fig1_preprocessing_pipeline.png", dpi=300, bbox_inches='tight')
plt.show()