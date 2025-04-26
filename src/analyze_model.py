import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# === Load Combined Dataset ===
df = pd.concat([
    pd.read_csv("/Users/harimanivannan/Documents/GitHub/music-classifier/data/mec/mec_features.csv"),
    pd.read_csv("/Users/harimanivannan/Documents/GitHub/music-classifier/data/deam/deam_features.csv"),
    pd.read_csv("/Users/harimanivannan/Documents/GitHub/music-classifier/data/emopia/emopia_features.csv")
], ignore_index=True)

# === Preprocess Data ===
df.dropna(inplace=True)

le = LabelEncoder()
df["emotion_label"] = le.fit_transform(df["emotion"])

X = df.drop(columns=["file", "emotion", "emotion_label"])
X = X.applymap(lambda x: float(x.strip("[]")) if isinstance(x, str) and x.startswith("[") else x)
y = df["emotion_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Load Voting Ensemble Model ===
ensemble = joblib.load("/Users/harimanivannan/Documents/GitHub/music-classifier/src/models/XGBoost.pkl")

# === Confusion Matrix ===
y_pred = ensemble.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.show()

# === Feature Distributions ===
# Add feature values back to the original DataFrame
X_df = pd.DataFrame(X, columns=X.columns if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])])
X_df["emotion"] = df["emotion"].values
df = pd.concat([df.reset_index(drop=True), X_df.reset_index(drop=True)], axis=1)

# Plot selected features by emotion
selected_features = ["tempo", "zcr_mean", "chroma_mean", "mfcc1_mean"]
for feature in selected_features:
    if feature in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="emotion", y=feature, data=df)
        plt.title(f"{feature} by Emotion")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
