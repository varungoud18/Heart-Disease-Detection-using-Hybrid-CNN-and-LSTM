import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from phase3_model import build_cnn_lstm_model

def main():
    print("=== Phase 4: Training & Optimization ===")
    
    # 1. Load Preprocessed Data
    print("Loading datasets...")
    X_train = np.load(r'd:\heart_disease\X_train.npy')
    y_train = np.load(r'd:\heart_disease\y_train.npy')
    # We will use 10% of train data as validation split to keep X_test pure for Phase 5
    
    # 2. Compute Class Weights
    # The normal class heavily outnumbers arrhythmia classes. Using class weights 
    # instead of SMOTE avoids generating "fake" ECG electrical signals and forces
    # the network to heavily penalize missing the minority classes.
    print("Computing class weights to handle severe class imbalance...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {c: w for c, w in zip(classes, weights)}
    
    print("Calculated Class Weights:")
    for c, w in class_weights_dict.items():
        print(f"  Class {c}: {w:.4f}")
        
    # 3. Model Import and Compilation
    model = build_cnn_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=5)
    
    # 4. Callbacks (Early Stopping)
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5,             # stop if validation loss doesn't improve for 5 epochs
        restore_best_weights=True,
        verbose=1
    )
    
    model_path = r'd:\heart_disease\heart_disease_cnn_lstm.h5'
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks_list = [early_stop, checkpoint]
    
    # 5. Training
    print("\nStarting Training Engine...")
    print("Training parameters: batch_size=256, max_epochs=30 (with EarlyStopping)")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,    # Using 10% of training data for Early Stopping
        epochs=30,               # High max epochs, relying on EarlyStopping to halt training
        batch_size=256,
        class_weight=class_weights_dict,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 6. Plotting the results
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='lightblue', linestyle='--')
    plt.title('Network Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    plt.plot(history.history['val_loss'], label='Val Loss', color='salmon', linestyle='--')
    plt.title('Network Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = r'd:\heart_disease\phase4_history.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training History Plot saved to '{plot_path}'.")
    print(f"\nTraining Complete! Best model saved locally as '{model_path}'.")

if __name__ == '__main__':
    main()
