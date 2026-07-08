import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GlobalAveragePooling1D, Dense, Dropout # type: ignore

def build_cnn_lstm_model(input_shape=(300, 1), num_classes=5):
    """
    Builds the Hybrid 1D-CNN + LSTM framework.
    """
    model = Sequential(name="Hybrid_CNN_LSTM")
    
    # --- 1. CNN Layers for Spatial Feature Extraction ---
    # Block 1
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape, name="Conv1D_1"))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool1D_1"))
    
    # Block 2
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', name="Conv1D_2"))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool1D_2"))
    
    # Block 3
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', name="Conv1D_3"))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool1D_3"))
    
    # --- 2. Temporal Layer ---
    # We must set return_sequences=True so that the GlobalAveragePooling1D can process the timeline.
    model.add(LSTM(64, return_sequences=True, name="LSTM_Layer"))
    
    # --- 3. Head ---
    model.add(GlobalAveragePooling1D(name="Global_Avg_Pool"))
    model.add(Dropout(0.3, name="Dropout")) # Add a small dropout to prevent overfitting
    model.add(Dense(num_classes, activation='softmax', name="Output_Classification"))
    
    # --- Compilation ---
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # using sparse because y is just an integer (0-4)
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("=== Phase 3: Hybrid CNN-LSTM Model Construction ===")
    
    # Create the model using the segment length from Phase 2 (300 samples)
    model = build_cnn_lstm_model(input_shape=(300, 1), num_classes=5)
    
    model.summary()
    print("\nModel Definition Logic:")
    print("- CNN Stack: Three 1D-Convolution layers (32->64->128 filters) smoothly shrink the temporal size while extracting local morphology (R-peaks, P-waves, etc.).")
    print("- LSTM Layer: A 64-unit LSTM block models the temporal dynamics and sequencing of the extracted spatial features.")
    print("- GAP & Head: Global Average Pooling maps the sequenced vectors globally, feeding into a 5-class Softmax.")
    print("- Optimizer/Loss: Adam optimizer with Sparse Categorical Crossentropy for our integer-based AAMI mapping.\n")
    print("Model construction complete. Ready for Phase 4 Training!")
