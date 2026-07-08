import os
import wfdb
import numpy as np
import scipy.signal as signal
from collections import Counter
from sklearn.model_selection import train_test_split

# AAMI Mapping Dictionary
aami_mapping = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', # Normal
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',           # Supraventricular
    'V': 'V', 'E': 'V',                               # Ventricular
    'F': 'F',                                         # Fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q'                      # Unknown
}

# Label to Int mapping
label_to_int = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
int_to_label = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

def apply_bandpass_filter(sig, fs=360):
    """Apply a digital bandpass filter 0.5 to 45 Hz."""
    lowcut = 0.5
    highcut = 45.0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 4th order Butterworth filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_sig = signal.filtfilt(b, a, sig)
    return filtered_sig

def zscore_normalize(segment):
    """Perform Z-score normalization."""
    mean = np.mean(segment)
    std = np.std(segment)
    if std == 0:
        return segment - mean
    return (segment - mean) / std

def main():
    dataset_path = r'd:\heart_disease\dataset\mit-bih-arrhythmia-database-1.0.0'
    print("=== Phase 2: Signal Preprocessing ===")
    
    records = set([f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.hea')])
    records = sorted(list(records))
    
    X_all = []
    y_all = []
    
    window_before = 150
    window_after = 150
    total_window = window_before + window_after
    
    print(f"Processing {len(records)} records...")
    
    for record_name in records:
        record_path = os.path.join(dataset_path, record_name)
        
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"Failed to load {record_name}: {e}")
            continue
            
        # We use MLII (typically channel 0)
        sig = record.p_signal[:, 0]
        fs = record.fs
        
        # 1. Noise Removal (Bandpass Filter)
        sig_filtered = apply_bandpass_filter(sig, fs=fs)
        
        labels = annotation.symbol
        samples = annotation.sample
        
        # 2. Segmentation & AAMI Mapping
        for symbol, sample in zip(labels, samples):
            # Check if symbol maps to one of our AAMI categories
            if symbol in aami_mapping:
                aami_class = aami_mapping[symbol]
                
                # Check boundaries to ensure we can extract a full window
                if sample >= window_before and sample < (len(sig_filtered) - window_after):
                    segment = sig_filtered[sample - window_before : sample + window_after]
                    
                    # 3. Normalization
                    segment_norm = zscore_normalize(segment)
                    
                    X_all.append(segment_norm)
                    y_all.append(label_to_int[aami_class])

    print("Data extraction complete.")
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Needs a 3D shape for 1D-CNN: (samples, time_steps, features)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))
    
    print(f"\nFinal extracted dataset shape - X: {X_all.shape}, y: {y_all.shape}")
    
    # Class distribution
    counts = Counter(y_all)
    print("\nClass distribution:")
    for int_label, count in sorted(counts.items()):
        print(f"  {int_to_label[int_label]} (Class {int_label}): {count} samples")
        
    # Create train and test split now to preserve independent sets for evaluation
    # Using 80% train and 20% test
    # (In formal academic studies, they often split by patient (record ID), but here we do random stratified split)
    print("\nSplitting into Train / Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Save the processed data for next phases so we don't have to re-compute
    np.save(r'd:\heart_disease\X_train.npy', X_train)
    np.save(r'd:\heart_disease\y_train.npy', y_train)
    np.save(r'd:\heart_disease\X_test.npy', X_test)
    np.save(r'd:\heart_disease\y_test.npy', y_test)
    
    print("\nSaved processed datasets as .npy files in d:\\heart_disease\\")

if __name__ == '__main__':
    main()
