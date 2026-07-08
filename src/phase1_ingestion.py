import os
import sys
import wfdb
import matplotlib.pyplot as plt

# Get project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    dataset_path = os.path.join(ROOT_DIR, 'dataset', 'mit-bih-arrhythmia-database-1.0.0')
    
    print("=== Phase 1: Data Ingestion & Directory Mapping ===")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    print(f"Loading dataset from: {dataset_path}")
    
    records = set([f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.hea')])
    records = sorted(list(records))
    print(f"Found {len(records)} records in standard PhysioNet format.\n")
    
    record_name = records[0]
    record_path = os.path.join(dataset_path, record_name)
    
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        sig = record.p_signal[:, 0]
        labels = annotation.symbol
        samples = annotation.sample
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return

    from collections import Counter
    class_counts = Counter(labels)
    print(f"--- Quick Peek into Record {record_name} ---")
    print(f"Signal Length: {len(sig)} samples, Sampling Rate: {record.fs} Hz")
    print("Annotation Types Count in this record:")
    for label, count in class_counts.items():
        print(f"  Type '{label}': {count}")
    
    normal_idx = None
    arrhythmia_idx = None
    
    for i, symbol in enumerate(labels):
        if symbol == 'N' and normal_idx is None:
            if samples[i] > 150 and samples[i] < len(sig) - 150:
                normal_idx = samples[i]
        elif symbol in ['V', 'S', 'A', 'F', 'R', 'L'] and arrhythmia_idx is None:
            if samples[i] > 150 and samples[i] < len(sig) - 150:
                arrhythmia_idx = samples[i]
                arrhythmia_type = symbol
                
        if normal_idx is not None and arrhythmia_idx is not None:
            break
            
    if normal_idx is None or arrhythmia_idx is None:
        print("Couldn't find both a Normal and Arrhythmia sample cleanly in this record.")
        return
        
    print(f"\nExtracted one Normal (N) at sample {normal_idx}")
    print(f"Extracted one Arrhythmia ({arrhythmia_type}) at sample {arrhythmia_idx}")
    
    window = 150 
    
    normal_signal = sig[normal_idx - window : normal_idx + window]
    arrhythmia_signal = sig[arrhythmia_idx - window : arrhythmia_idx + window]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(normal_signal, color='blue')
    plt.title("Normal Heartbeat (N)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(arrhythmia_signal, color='red')
    plt.title(f"Arrhythmia Heartbeat ({arrhythmia_type})")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(ROOT_DIR, 'overleaf_images', 'phase1_plot.png')
    
    # Ensure overleaf_images directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    plt.savefig(plot_path)
    print(f"\nSaved comparison plot to '{plot_path}'.")
    plt.show()

if __name__ == '__main__':
    main()
