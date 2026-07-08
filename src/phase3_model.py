import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GlobalAveragePooling1D, Dense, Dropout # type: ignore

def build_cnn_lstm_model(input_shape=(300, 1), num_classes=5):
    """
    Builds the Hybrid 1D-CNN + LSTM framework.
    """
    model = Sequential(name="Hybrid_CNN_LSTM")
    
    # Block 1
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape, name="Conv1D_1"))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool1D_1"))
    
    # Block 2
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', name="Conv1D_2"))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool1D_2"))
    
    # Block 3
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', name="Conv1D_3"))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool1D_3"))
    
    # Temporal Layer
    model.add(LSTM(64, return_sequences=True, name="LSTM_Layer"))
    
    # Head
    model.add(GlobalAveragePooling1D(name="Global_Avg_Pool"))
    model.add(Dropout(0.3, name="Dropout"))
    model.add(Dense(num_classes, activation='softmax', name="Output_Classification"))
    
    # Compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("=== Phase 3: Hybrid CNN-LSTM Model Construction ===")
    model = build_cnn_lstm_model(input_shape=(300, 1), num_classes=5)
    model.summary()
