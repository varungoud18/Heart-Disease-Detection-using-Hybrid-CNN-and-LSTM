import cv2
import numpy as np

def extract_ecg_from_image(image_path, target_length=300):
    """
    Reads an ECG image, extracts the line trace, and converts it into a 
    1D normalized numerical array of length `target_length` (default 300)
    suitable for the CNN-LSTM model.
    """
    # 1. Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}. Please check the path and ensure it is a valid image file.")
    
    height, width = img.shape
    
    # 2. Thresholding
    # ECG traces are usually dark lines on a lighter grid/background.
    # cv2.THRESH_BINARY_INV makes the dark lines white (255) and the background black (0).
    # You might need to adjust the threshold value (100) depending on image contrast.
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Extract the trace (getting the Y-coordinate for each X-coordinate)
    signal = []
    x_indices = []
    
    for x in range(width):
        col = thresh[:, x]
        y_indices = np.where(col > 0)[0]
        
        if len(y_indices) > 0:
            # If the trace is highly thick, take the mean Y point of the trace
            # In image coords, Y=0 is TOP. To make it a standard graph where 
            # higher value is "up", we subtract the Y from height.
            y_val = height - np.mean(y_indices)
            signal.append(y_val)
            x_indices.append(x)
            
    if not signal:
        raise ValueError("Could not find any clear trace lines in the image. The image might be too bright or lack a distinct dark ECG line.")
        
    # Convert to numpy arrays
    x_coords = np.array(x_indices)
    y_coords = np.array(signal)
    
    # 4. Interpolate over missing columns (if any) and resample to precisely target_length (300)
    # We span the original min to max, generating target_length equidistant points
    target_x = np.linspace(x_coords.min(), x_coords.max(), target_length)
    resampled_signal = np.interp(target_x, x_coords, y_coords)
    
    # 5. Z-Score Normalization
    # The models are trained on normalized data (mean=0, std=1)
    mean_val = np.mean(resampled_signal)
    std_val = np.std(resampled_signal)
    
    if std_val == 0:
        # Prevent division by zero if it's a flatline
        normalized_signal = resampled_signal - mean_val
    else:
        normalized_signal = (resampled_signal - mean_val) / std_val
        
    return target_x, normalized_signal, img
