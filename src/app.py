import streamlit as st
import os
import joblib
import librosa
import numpy as np
import pandas as pd
import tempfile
from pydub import AudioSegment
import pretty_midi

st.set_page_config(page_title="Musical Emotion Classifier", layout="centered")

st.title(" Musical Emotion Classifier")
st.markdown("Upload a `.wav`, `.mp3`, or `.mid` audio file to predict the emotion.")

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
def convert_to_wav(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    if ext == ".wav":
        temp_file.write(uploaded_file.read())
        return temp_file.name

    elif ext == ".mp3":
        audio = AudioSegment.from_file(uploaded_file, format="mp3")
        audio.export(temp_file.name, format="wav")
        return temp_file.name

    elif ext in [".mid", ".midi"]:
        midi = pretty_midi.PrettyMIDI(uploaded_file)
        audio = midi.synthesize()
        librosa.output.write_wav(temp_file.name, audio, sr=22050)
        return temp_file.name

    else:
        return None

# ======= PREDICTION LOGIC =======
def predict_emotion(wav_path):
    feats = extract_features(wav_path)
    X = pd.DataFrame([feats])

    # Load preprocessing tools and model
    scaler = joblib.load("/Users/harimanivannan/Documents/GitHub/music-classifier/src/models/scaler.pkl")
    label_encoder = joblib.load("/Users/harimanivannan/Documents/GitHub/music-classifier/src/models/label_encoder.pkl")
    expected_columns = scaler.feature_names_in_

    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[expected_columns]

    X_scaled = scaler.transform(X)
    model = joblib.load("/Users/harimanivannan/Documents/GitHub/music-classifier/src/models/XGBoost.pkl")
    y_pred = model.predict(X_scaled)
    return label_encoder.inverse_transform(y_pred)[0]

# ======= UI =======
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "mid"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing the audio..."):
        wav_path = convert_to_wav(uploaded_file)
        if not wav_path:
            st.error("Unsupported audio format.")
        else:
            try:
                emotion = predict_emotion(wav_path)
                st.success(f"Predicted Emotion: **{emotion.capitalize()}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
