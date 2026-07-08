import os
import sys
import numpy as np
import pytest
import tensorflow as tf

# Ensure root directory is in python path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase2_preprocessing import apply_bandpass_filter, zscore_normalize, aami_mapping, label_to_int
from phase3_model import build_cnn_lstm_model
from inference import predict_heartbeat, int_to_label

def test_aami_mapping():
    """Verify that clinical labels map correctly to AAMI categories."""
    assert aami_mapping['N'] == 'N'
    assert aami_mapping['L'] == 'N'
    assert aami_mapping['R'] == 'N'
    assert aami_mapping['V'] == 'V'
    assert aami_mapping['A'] == 'S'
    assert aami_mapping['F'] == 'F'
    assert aami_mapping['/'] == 'Q'
    assert label_to_int['N'] == 0
    assert label_to_int['Q'] == 4

def test_bandpass_filter():
    """Test the Butterworth bandpass filter output length and basic behavior."""
    fs = 360
    t = np.linspace(0, 1, fs)
    # 2 Hz signal (should pass) + 60 Hz line noise (should be attenuated)
    clean_sig = np.sin(2 * np.pi * 2 * t)
    noise_sig = np.sin(2 * np.pi * 60 * t)
    noisy_sig = clean_sig + noise_sig
    
    filtered = apply_bandpass_filter(noisy_sig, fs=fs)
    
    assert len(filtered) == len(noisy_sig)
    # Check that high frequency noise is significantly attenuated
    # Standard deviation of high freq noise alone is approx 0.707
    # Standard deviation of difference between filtered and clean should be small
    diff_noise = filtered - clean_sig
    assert np.std(diff_noise) < np.std(noise_sig)

def test_zscore_normalize():
    """Test that Z-score normalization produces mean=0 and std=1."""
    np.random.seed(42)
    segment = np.random.normal(loc=5.0, scale=2.0, size=300)
    
    normalized = zscore_normalize(segment)
    
    assert len(normalized) == 300
    assert np.isclose(np.mean(normalized), 0.0, atol=1e-7)
    assert np.isclose(np.std(normalized), 1.0, atol=1e-7)

def test_flatline_normalize():
    """Test that Z-score normalization handles flatlines (all same values) gracefully."""
    flatline = np.ones(300) * 1.5
    normalized = zscore_normalize(flatline)
    
    assert len(normalized) == 300
    assert np.all(normalized == 0.0)

def test_cnn_lstm_model_compilation():
    """Test model creation and architecture configurations."""
    input_shape = (300, 1)
    num_classes = 5
    model = build_cnn_lstm_model(input_shape=input_shape, num_classes=num_classes)
    
    assert model.name == "Hybrid_CNN_LSTM"
    assert model.input_shape == (None, 300, 1)
    assert model.output_shape == (None, 5)
    
    # Check key layers exist
    layers_names = [layer.name for layer in model.layers]
    assert any("Conv1D" in name for name in layers_names)
    assert any("LSTM" in name for name in layers_names)
    assert any("Output_Classification" in name for name in layers_names)

def test_predict_heartbeat_mapping():
    """Test predict_heartbeat function handles mock prediction outputs properly."""
    # Create a mock model class
    class MockModel:
        def predict(self, x, verbose=0):
            # Output class probabilities. Let's make Ventricular (class 2) the highest.
            probs = np.zeros((1, 5))
            probs[0, 2] = 0.95
            probs[0, 0] = 0.05
            return probs
            
    mock_model = MockModel()
    dummy_signal = np.zeros(300)
    
    diagnosis, confidence, pred_class = predict_heartbeat(mock_model, dummy_signal)
    
    assert pred_class == 2
    assert np.isclose(confidence, 95.0)
    assert diagnosis == "HEART DISEASE DETECTED (Arrhythmia Type: V (Ventricular))"
