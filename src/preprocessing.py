import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

# === Paths ===
BASE = "/Users/harimanivannan/Documents/GitHub/music-classifier/data"
OUT_FILE = os.path.join(BASE, "processed_dataset.csv")

# === Load Turkish Music Emotion ===
turkish_path = os.path.join(BASE, "a_f.csv")
turkish_df = pd.read_csv(turkish_path)
turkish_df = turkish_df.rename(columns={"Class": "emotion"})
turkish_df["emotion"] = turkish_df["emotion"].str.lower()
turkish_df["source"] = "turkish"

# === Load DEAM ===
deam_ann1 = pd.read_csv(os.path.join(BASE, "deam", "static_annotations_averaged_songs_1_2000.csv"))
deam_ann2 = pd.read_csv(os.path.join(BASE, "deam", "static_annotations_averaged_songs_2000_2058.csv"))
deam_df = pd.concat([deam_ann1, deam_ann2], ignore_index=True)
deam_df.columns = deam_df.columns.str.strip()

def classify_deam(valence, arousal):
    if valence > 5 and arousal > 5:
        return "happy"
    elif valence > 5 and arousal <= 5:
        return "calm"
    elif valence <= 5 and arousal > 5:
        return "angry"
    else:
        return "sad"

deam_df["emotion"] = deam_df.apply(lambda x: classify_deam(x["valence_mean"], x["arousal_mean"]), axis=1)
deam_df["source"] = "deam"

# === Load Musical Emotions Classification ===
music_train = pd.read_csv(os.path.join(BASE, "mec", "train.csv"))
music_test = pd.read_csv(os.path.join(BASE, "mec", "test.csv"))
music_df = pd.concat([music_train, music_test], ignore_index=True)
music_df = music_df.rename(columns={"Target": "emotion"})
music_df["emotion"] = music_df["emotion"].str.lower()
music_df["source"] = "musical_emotions"

# === Load EMOPIA ===
emopia_meta = pd.read_csv(os.path.join(BASE, "emopia", "metadata_by_song.csv"))
quad_map = {1: "happy", 2: "angry", 3: "sad", 4: "calm"}
emopia_meta["emotion"] = emopia_meta["DominantQ"].map(quad_map)
emopia_meta["source"] = "emopia"

# === Feature Selection from Each Dataset ===
def select_features(df):
    feature_cols = [
        col for col in df.columns
        if any(x in col for x in ["tempo", "loudness", "pitch", "Spectral", "MFCC", "ZeroCrossing", "Harmonic", "Mean"])
    ]
    features = df[feature_cols].copy()
    features["emotion"] = df["emotion"]
    features["source"] = df["source"]
    return features

turkish_out = select_features(turkish_df)
deam_out = select_features(deam_df)
music_out = music_df[["emotion", "source"]]  # no acoustic features
emopia_out = emopia_meta[["emotion", "source"]]  # no acoustic features

# === Combine All ===
full_df = pd.concat([turkish_out, deam_out, music_out, emopia_out], ignore_index=True)

# === Normalize Numeric Features ===
numeric_cols = full_df.select_dtypes(include=["float64", "int64"]).columns
scaler = MinMaxScaler()
full_df[numeric_cols] = scaler.fit_transform(full_df[numeric_cols])

# === Drop Rows with Missing Values ===
full_df_clean = full_df.dropna(subset=numeric_cols)

# === Variance Threshold ===
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(full_df_clean[numeric_cols])
var_selected_cols = full_df_clean[numeric_cols].columns[selector.get_support()]
df_reduced = full_df_clean[var_selected_cols.tolist() + ["emotion", "source"]]

# === Correlation Filter ===
corr_matrix = df_reduced[var_selected_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
df_reduced = df_reduced.drop(columns=to_drop)

# === Top Feature Selection by Random Forest ===
X = df_reduced.drop(columns=["emotion", "source"])
y = df_reduced["emotion"]
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(25).index.tolist()

# === Final Dataset ===
final_df = df_reduced[top_features + ["emotion", "source"]]
final_df.to_csv(OUT_FILE, index=False)
print(f"✅ Preprocessing complete. Processed dataset saved to: {OUT_FILE}")
