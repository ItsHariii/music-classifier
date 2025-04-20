import pandas as pd
import os

# Base path to your dataset folders
base_path = "data"

datasets = {
    "Turkish Music Emotion": os.path.join(base_path, "Acoustic Features.csv"),
    "DEAM - Arousal": os.path.join(base_path, "deam", "arousal.csv"),
    "DEAM - Valence": os.path.join(base_path, "deam", "valence.csv"),
    "DEAM - Annotations Part 1": os.path.join(base_path, "deam", "static_annotations_averaged_songs_1_2000.csv"),
    "DEAM - Annotations Part 2": os.path.join(base_path, "deam", "static_annotations_averaged_songs_2000_2058.csv"),
    "Musical Emotions - Train": os.path.join(base_path, "mec", "train.csv"),
    "Musical Emotions - Test": os.path.join(base_path, "mec", "test.csv"),
    "EMOPIA - Labels": os.path.join(base_path, "emopia", "label.csv"),
    "EMOPIA - Metadata": os.path.join(base_path, "emopia", "metadata_by_song.csv"),
}

# Print the first 5 rows of each dataset
for name, path in datasets.items():
    try:
        print(f"\n=== {name} ===")
        df = pd.read_csv(path)
        print(df.head())
    except Exception as e:
        print(f"Could not load {name}: {e}")