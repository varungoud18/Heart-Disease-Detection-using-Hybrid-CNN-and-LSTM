import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Get project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.phase3_model import build_cnn_lstm_model

def main():
    print("=== Phase 4: Training & Optimization ===")
    
    X_train_path = os.path.join(ROOT_DIR, 'data', 'X_train.npy')
    y_train_path = os.path.join(ROOT_DIR, 'data', 'y_train.npy')
    
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print(f"Error: Preprocessed data files not found in '{os.path.dirname(X_train_path)}'. Please run Phase 2.")
        return
        
    print("Loading datasets...")
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    
    print("Computing class weights to handle severe class imbalance...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {c: w for c, w in zip(classes, weights)}
    
    print("Calculated Class Weights:")
    for c, w in class_weights_dict.items():
        print(f"  Class {c}: {w:.4f}")
        
    model = build_cnn_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=5)
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    models_dir = os.path.join(ROOT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'heart_disease_cnn_lstm.h5')
    
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks_list = [early_stop, checkpoint]
    
    print("\nStarting Training Engine...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=256,
        class_weight=class_weights_dict,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='lightblue', linestyle='--')
    plt.title('Network Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    plt.plot(history.history['val_loss'], label='Val Loss', color='salmon', linestyle='--')
    plt.title('Network Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(ROOT_DIR, 'overleaf_images', 'phase4_history.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training History Plot saved to '{plot_path}'.")
    print(f"\nTraining Complete! Best model saved locally as '{model_path}'.")

if __name__ == '__main__':
    main()
