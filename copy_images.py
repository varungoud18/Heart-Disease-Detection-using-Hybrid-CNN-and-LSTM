import os
import shutil

src = r'd:\heart_disease'
dst = os.path.join(src, 'overleaf_images')
os.makedirs(dst, exist_ok=True)

files = [
    'phase1_plot.png',
    'fig1_preprocessing_pipeline.png',
    'fig2_cnn_lstm_architecture.png',
    'fig3_vision_inference.png',
    'phase4_history.png',
    'fig4_confusion_matrix.png',
    'confusion_matrix.png',
    'fig5_performance_bars.png',
    'inference_plot.png',
]

for f in files:
    src_path = os.path.join(src, f)
    dst_path = os.path.join(dst, f)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {f}")
    else:
        print(f"NOT FOUND: {f}")

print(f"\nAll images copied to: {dst}")
print(f"Total files in folder: {len(os.listdir(dst))}")
