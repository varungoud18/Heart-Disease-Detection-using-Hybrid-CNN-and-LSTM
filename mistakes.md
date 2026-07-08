# Analysis of Vulnerabilities and Mistakes in Inference Pipeline

Based on the investigation into your `inference.py` engine and the underlying model pipeline, I have identified several critical mistakes and structural flaws that explain why you are consistently getting a "No Heart Disease Detected" result, despite having images of classified arrhythmias.

### 1. Zero Image Processing Capabilities
The most critical issue is that **your model and inference script do not process images at all**.
You mentioned that "no matter the input image" the results are the same. However, `inference.py` does not contain any code (like OpenCV or PIL) to open `.jpg`, `.png`, or `.webp` files. 
The script and the Neural Network (`CNN-LSTM`) are strictly designed to ingest **1D numerical time-series arrays** (shape `(300,)`) representing electrical voltage readings, which were preprocessed from the physical MIT-BIH `wfdb` files. Any images placed in your directory are completely ignored by the execution logic. 

### 2. Random Execution Instead of Provided Input
If you review `inference.py`, the code does not accept commands, filenames, or arguments. Instead, it bypasses your intentions completely by hardcoding a random number generator:
```python
# Pick a random sample index
random_idx = np.random.randint(0, len(X_test))
patient_signal = X_test[random_idx]
```
Even if your system could process images, the code as currently written simply picks a random row from the gigabytes of array data in `X_test.npy` every time you run it.

### 3. Statistical Disproportion (The MIT-BIH Skew)
The reason it almost always outputs "No Heart Disease Detected" (Class 0: Normal) is tied directly to the random selection mentioned above. The `X_test.npy` dataset is derived from the MIT-BIH dataset, which is overwhelmingly composed of normal heartbeats (~80-85%). 
Because the code pulls a random heartbeat from this test distribution, statistical probability dictates that roughly 8 or 9 out of 10 times you run the program, it randomly picks a completely healthy heartbeat line, leading to the constant "Normal" outputs you're experiencing.

### Summary
You are essentially running a lottery system that pulls a pre-existing numerical graph mostly consisting of normal rhythms, while mistakenly believing the system is evaluating the local `.jpg/.png` files you are providing.

**Next Steps (Conceptual):**
To fix this, you would need to:
1. Write a Computer Vision script that can scan user-provided ECG images and extract the ink lines into a 1D numerical array.
2. Alter `inference.py` to accept local file argument paths (e.g., `python inference.py --image my_ecg.jpg`) rather than using `np.random`.
