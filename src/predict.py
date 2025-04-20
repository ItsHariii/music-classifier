import sys
import os
import joblib
import librosa
import numpy as np
import pandas as pd
import tempfile
from pydub import AudioSegment
import pretty_midi

# ======= FEATURE EXTRACTION =======
def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        features[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc{i+1}_std"] = np.std(mfcc[i])

    # Delta MFCC
    delta_mfcc = librosa.feature.delta(mfcc)
    for i in range(delta_mfcc.shape[0]):
        features[f"delta_mfcc{i+1}_mean"] = np.mean(delta_mfcc[i])
        features[f"delta_mfcc{i+1}_std"] = np.std(delta_mfcc[i])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["contrast_mean"] = np.mean(contrast)
    features["contrast_std"] = np.std(contrast)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["bandwidth_mean"] = np.mean(bandwidth)
    features["bandwidth_std"] = np.std(bandwidth)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"] = np.mean(rolloff)
    features["rolloff_std"] = np.std(rolloff)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(tonnetz.shape[0]):
        features[f"tonnetz{i+1}_mean"] = np.mean(tonnetz[i])
        features[f"tonnetz{i+1}_std"] = np.std(tonnetz[i])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = np.mean(zcr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = tempo

    return features

# ======= FILE CONVERSION =======
def convert_to_wav(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path
    elif ext == ".mp3":
        audio = AudioSegment.from_mp3(input_path)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name
    elif ext in [".mid", ".midi"]:
        midi = pretty_midi.PrettyMIDI(input_path)
        audio = midi.synthesize()
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        librosa.output.write_wav(temp_wav.name, audio, sr=22050)
        return temp_wav.name
    else:
        raise ValueError("Unsupported audio format")

# ======= MAIN PREDICTOR =======
if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/audio.[wav|mp3|mid]")
    sys.exit(1)

input_path = sys.argv[1]
if not os.path.isfile(input_path):
    print(f"[Error] File not found: {input_path}")
    sys.exit(1)

try:
    wav_path = convert_to_wav(input_path)
    feats = extract_features(wav_path)
    X = pd.DataFrame([feats])

    # Load scaler, label encoder, and expected columns
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    expected_columns = scaler.feature_names_in_

    # Ensure all expected features are present in the right order
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0.0  # Fill missing columns
    X = X[expected_columns]  # Reorder

    X_scaled = scaler.transform(X)

    # Load ensemble model
    model = joblib.load("models/VotingEnsemble.pkl")
    y_pred = model.predict(X_scaled)
    emotion = label_encoder.inverse_transform(y_pred)[0]
    print(f"[Prediction] Emotion: {emotion}")
except Exception as e:
    print(f"[Error] {e}")
