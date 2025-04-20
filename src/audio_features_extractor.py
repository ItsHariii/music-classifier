import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# === CONFIGURATION ===
datasets = [
    {
        "name": "mec",
        "audio_dir": "/Users/harimanivannan/Documents/GitHub/music-classifier/data/mec/audio",
        "labels_csv": None,
        "output_csv": "/Users/harimanivannan/Documents/GitHub/music-classifier/data/mec/mec_features.csv"
    },
    {
        "name": "deam",
        "audio_dir": "/Users/harimanivannan/Documents/GitHub/music-classifier/data/deam/audio_wav",
        "labels_csv": [
            "/Users/harimanivannan/Documents/GitHub/music-classifier/data/deam/static_annotations_averaged_songs_1_2000.csv",
            "/Users/harimanivannan/Documents/GitHub/music-classifier/data/deam/static_annotations_averaged_songs_2000_2058.csv"
        ],
        "output_csv": "/Users/harimanivannan/Documents/GitHub/music-classifier/data/deam/deam_features.csv"
    },
    {
        "name": "emopia",
        "audio_dir": "/Users/harimanivannan/Documents/GitHub/music-classifier/data/emopia/audio_wav",
        "labels_csv": "/Users/harimanivannan/Documents/GitHub/music-classifier/data/emopia/metadata_by_song.csv",
        "output_csv": "/Users/harimanivannan/Documents/GitHub/music-classifier/data/emopia/emopia_features.csv"
    }
]

def process_dataset(audio_root, label_lookup, output_csv):
    all_data = []
    for rel_path, label in tqdm(label_lookup.items()):
        file_path = os.path.join(audio_root, rel_path)
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {rel_path}")
            continue
        try:
            feats = extract_features(file_path)
            feats["file"] = rel_path
            feats["emotion"] = label
            all_data.append(feats)
        except Exception as e:
            print(f"⚠️ Error processing {rel_path}: {e}")
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv}")


# === RUN EXTRACTION ===
for ds in datasets:

    print(f"\n🎧 Processing {ds['name']}...")
    
    if ds["name"] == "mec":
        label_map = {}
        for subset in ["train", "test"]:
            base_path = os.path.join(ds["audio_dir"], subset)
            for emotion in os.listdir(base_path):
                emotion_folder = os.path.join(base_path, emotion)
                if os.path.isdir(emotion_folder):
                    for file in os.listdir(emotion_folder):
                        if file.endswith(".wav"):
                            # Save relative path to the file
                            rel_path = os.path.join(subset, emotion, file)
                            label_map[rel_path] = emotion.lower()

    elif ds["name"] == "deam":
        # Merge both DEAM label CSVs
        if isinstance(ds["labels_csv"], list):
            labels = pd.concat([pd.read_csv(f) for f in ds["labels_csv"]], ignore_index=True)
        else:
            labels = pd.read_csv(ds["labels_csv"])

        # Clean column names
        labels.columns = labels.columns.str.strip()

        # Compute emotion labels from valence & arousal
        label_map = dict(zip(
            labels["song_id"].astype(str) + ".wav",
            labels.apply(
                lambda x: "happy" if x["valence_mean"] > 5 and x["arousal_mean"] > 5
                else "calm" if x["valence_mean"] > 5
                else "angry" if x["arousal_mean"] > 5
                else "sad",
                axis=1
            )
        ))

    elif ds["name"] == "emopia":
        labels = pd.read_csv(ds["labels_csv"])
        qmap = {1: "happy", 2: "angry", 3: "sad", 4: "calm"}
        
        # Match all wav files that contain the songID (partial match)
        label_map = {}
        for _, row in labels.iterrows():
            sid = row["songID"]
            label = qmap.get(row["DominantQ"], "unknown")
            for fname in os.listdir(ds["audio_dir"]):
                if fname.endswith(".wav") and sid in fname:
                    label_map[fname] = label

    process_dataset(ds["audio_dir"], label_map, ds["output_csv"])

