import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Get project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.extract_signal import extract_ecg_from_image

int_to_label = {
    0: 'N (Normal)', 
    1: 'S (Supraventricular)', 
    2: 'V (Ventricular)', 
    3: 'F (Fusion)', 
    4: 'Q (Unknown)'
}

def predict_heartbeat(model, segment_300_samples):
    """
    Takes a preprocessed 300-sample ECG window and uses the model 
    to decide if it implies heart disease (arrhythmia).
    """
    if segment_300_samples.ndim == 1:
        segment = segment_300_samples.reshape(1, 300, 1)
    elif segment_300_samples.ndim == 2:
        segment = segment_300_samples.reshape(1, 300, 1)
    else:
        segment = segment_300_samples
        
    prediction_probs = model.predict(segment, verbose=0)
    predicted_class_int = np.argmax(prediction_probs, axis=1)[0]
    confidence = np.max(prediction_probs) * 100
    class_name = int_to_label[predicted_class_int]
    
    if predicted_class_int == 0:
        diagnosis = "No Heart Disease Detected (Normal Rhythm)"
    else:
        diagnosis = f"HEART DISEASE DETECTED (Arrhythmia Type: {class_name})"
        
    return diagnosis, confidence, predicted_class_int

def main():
    parser = argparse.ArgumentParser(description="Heart Disease Inference Engine")
    parser.add_argument('--image', type=str, help="Path to the ECG image file (e.g. .jpg, .png)")
    parser.add_argument('--random', action='store_true', help="Pick a random numerical sample from the dataset if no image is provided")
    args = parser.parse_args()

    print("=== Heart Disease Inference Engine ===")
    
    model_path = os.path.join(ROOT_DIR, 'models', 'heart_disease_cnn_lstm.h5')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}! Ensure you completed Phase 4.")
        return
        
    print("Loading AI Model...")
    model = tf.keras.models.load_model(model_path)
    
    original_img = None
    true_label_str = "Unknown (Image Input)"

    if args.image:
        print(f"\nProcessing provided ECG Image: {args.image}")
        if not os.path.exists(args.image):
            print(f"Error: The file {args.image} does not exist.")
            return
            
        try:
            _, patient_signal, original_img = extract_ecg_from_image(args.image)
            print("Successfully digitized ECG trace from image.")
        except Exception as e:
            print(f"Error extracting signal from image: {e}")
            return
    elif args.random:
        print("Loading a random unseen patient ECG array sample...")
        X_test_path = os.path.join(ROOT_DIR, 'data', 'X_test.npy')
        y_test_path = os.path.join(ROOT_DIR, 'data', 'y_test.npy')
        
        if not os.path.exists(X_test_path):
            print(f"Cannot find {X_test_path}. Random inference requires the testing dataset.")
            return

        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        random_idx = np.random.randint(0, len(X_test))
        patient_signal = X_test[random_idx]
        true_label_str = int_to_label[y_test[random_idx]]
    else:
        print("No input provided. Please use '--image <path>' or add '--random' to use dataset.")
        return
    
    print("\nRunning Diagnostics...")
    diagnosis, confidence, pred_class = predict_heartbeat(model, patient_signal)
    
    print("\n--- Diagnostic Report ---")
    print(f"Algorithm Diagnosis: {diagnosis}")
    print(f"AI Confidence:       {confidence:.2f}%")
    print(f"True Ground Truth:   {true_label_str}")
    print("-------------------------\n")
    
    if args.image and original_img is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title(f"Original User Image\nFile: {args.image}")
        axes[0].axis('off')
        
        axes[1].plot(patient_signal.flatten(), color='green' if pred_class == 0 else 'red')
        axes[1].set_title(f"Digitized ECG Scan sent to AI\n{diagnosis} (Conf: {confidence:.2f}%)")
        axes[1].set_xlabel('Resampled Data Points')
        axes[1].set_ylabel('Amplitude (Z-score Normalized)')
        axes[1].grid(True)
    else:
        plt.figure(figsize=(8, 4))
        plt.plot(patient_signal.flatten(), color='green' if pred_class == 0 else 'red')
        plt.title(f"ECG Heartbeat Scan \n{diagnosis}")
        plt.xlabel('Samples / Timeline')
        plt.ylabel('Amplitude (Z-score)')
        plt.grid(True)
        
    plt.tight_layout()
    plot_path = os.path.join(ROOT_DIR, 'overleaf_images', 'inference_plot.png')
    plt.savefig(plot_path)
    print(f"Saved the visual scan of this heartbeat to '{plot_path}'.")

if __name__ == '__main__':
    main()
