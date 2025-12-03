import librosa
import numpy as np
import joblib
import os

# dynamically alligns filepath so future machines dont break it
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
model = joblib.load(MODEL_PATH)

REGION_TO_COORDS = {
    "Leinster":  (53.3498, -6.2603),
    "Munster":   (52.0667, -8.6333),
    "Connacht":  (53.2707, -9.0568),
    "Ulster":    (54.5970, -5.9300),
}

def extract_mfcc(path):
    # y = audio data (array of amplitudes).  sr = sample rate
    y, sr = librosa.load(path, sr=16000, mono=True)
    # Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)
    # “Because audio clips vary in length, I average the MFCCs over time so every clip becomes a fixed-length feature vector.”

def predict(path):
    # 2d array for scikit
    features = extract_mfcc(path).reshape(1, -1)
    label = model.predict(features)[0]           # e.g. "Leinster"
    coords = REGION_TO_COORDS.get(label, (53.4, -8.2))
    confidence = 0.75
    return coords, confidence, label
