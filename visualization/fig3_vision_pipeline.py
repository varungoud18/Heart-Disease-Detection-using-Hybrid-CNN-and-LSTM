import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Get project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
titles = ["(a) Input ECG Chart", "(b) Contour Extraction", "(c) Reconstructed Signal"]

ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
ax.add_patch(patches.Rectangle((1,1), 8, 4, facecolor='ivory', edgecolor='black'))
ax.plot([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
        [3.5, 3.2, 3.0, 2.8, 2.5, 2.2, 2.7, 3.5, 2.2, 1.8, 2.2, 3.0, 3.5, 3.2, 3.0], 'k-', lw=1.5)
ax.text(5, 0.5, "Digitized ECG tracing", ha='center', fontsize=8, style='italic')
ax.set_title(titles[0])

ax = axes[1]
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
ax.add_patch(patches.Rectangle((1,1), 8, 4, facecolor='black', edgecolor='white'))
ax.plot([1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5],
        [3.5,3.2,3.0,2.8,2.5,2.2,2.7,3.5,2.2,1.8,2.2,3.0,3.5,3.2,3.0], 'lime', lw=2)
ax.text(5, 0.5, "Largest contour (lime)", ha='center', fontsize=8, style='italic', color='white')
ax.set_title(titles[1])

ax = axes[2]
t = np.linspace(0, 1, 300)
signal = 0.6*np.sin(2*np.pi*5*t) + 0.3*np.sin(2*np.pi*12*t) + 0.1*np.random.randn(300)
ax.plot(t, signal, 'b-', lw=1)
ax.set_xlabel("Sample (time)")
ax.set_ylabel("Normalized amplitude")
ax.grid(True, alpha=0.3)
ax.set_title(titles[2])

plt.suptitle("Fig. 3: Vision-based inference engine – from printed ECG to 1D waveform", fontsize=10)
plt.tight_layout()

plot_path = os.path.join(ROOT_DIR, 'overleaf_images', 'fig3_vision_inference.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved: {plot_path}")
plt.show()
