import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    print("=== Phase 5: Evaluation & Research Benchmark ===")
    
    X_test_path = os.path.join(ROOT_DIR, 'data', 'X_test.npy')
    y_test_path = os.path.join(ROOT_DIR, 'data', 'y_test.npy')
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Error: Holdout test datasets not found at '{os.path.dirname(X_test_path)}'. Run Phase 2 first.")
        return
        
    print("Loading test datasets...")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    model_path = os.path.join(ROOT_DIR, 'models', 'heart_disease_cnn_lstm.h5')
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return
        
    print("Loading trained Hybrid CNN-LSTM network...")
    model = tf.keras.models.load_model(model_path)
    
    print("\nRunning test evaluations on 'unseen' data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"-> Base Test Accuracy: {test_acc * 100:.2f}%")
    print(f"-> Base Test Loss: {test_loss:.4f}\n")
    
    yildirim_target = 99.0
    print("=== Benchmark Analysis ===")
    print(f"Target Accuracy (Yildirim 2020): {yildirim_target}%")
    gap = test_acc * 100 - yildirim_target
    if gap >= 0:
        print(f"Result: EXCEEDED benchmark by {abs(gap):.2f}%! Excellent performance.")
    elif gap >= -3.0:
        print(f"Result: WITHIN margin of error to benchmark ({gap:.2f}%). Great performance.")
    else:
        print(f"Result: BELOW benchmark by {abs(gap):.2f}%. We might need heavier regularization or deeper models.")
        
    print("\nGenerating Detailed Metrics...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    labels_int = [0, 1, 2, 3, 4]
    target_names = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)', 'Q (Unknown)']
    
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, labels=labels_int, target_names=target_names))
    
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=labels_int)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Performance Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    plot_path = os.path.join(ROOT_DIR, 'overleaf_images', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Confusion Matrix Plot saved to '{plot_path}'.")
    print("\nPhase 5 Complete! The project is historically finalized.")

if __name__ == '__main__':
    main()
