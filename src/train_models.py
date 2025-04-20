import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.utils import resample

# === Load and Combine Feature CSVs ===
print("🔗 Loading features...")
dfs = []
for name in ["mec", "deam", "emopia"]:
    path = f"/Users/harimanivannan/Documents/GitHub/music-classifier/data/{name}/{name}_features.csv"
    df = pd.read_csv(path)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)


# Drop missing data
df.dropna(inplace=True)

# Print class distribution before balancing
print("📊 Original emotion distribution:")
print(df["emotion"].value_counts())

# Split by emotion
sad_df    = df[df["emotion"] == "sad"]
happy_df  = df[df["emotion"] == "happy"]
angry_df  = df[df["emotion"] == "angry"]
calm_df   = df[df["emotion"] == "calm"]

# Choose majority class size
target_size = max(len(sad_df), len(happy_df), len(angry_df), len(calm_df))

# Upsample each class
sad_up    = resample(sad_df,    replace=True, n_samples=target_size, random_state=42)
happy_up  = resample(happy_df,  replace=True, n_samples=target_size, random_state=42)
angry_up  = resample(angry_df,  replace=True, n_samples=target_size, random_state=42)
calm_up   = resample(calm_df,   replace=True, n_samples=target_size, random_state=42)

# Combine and shuffle
df = pd.concat([sad_up, happy_up, angry_up, calm_up]).sample(frac=1, random_state=42).reset_index(drop=True)

# Print new class distribution
print("✅ Balanced emotion distribution:")
print(df["emotion"].value_counts())

# Encode labels
le = LabelEncoder()
df["emotion_label"] = le.fit_transform(df["emotion"])


X = df.drop(columns=["file", "emotion", "emotion_label"])
X = X.applymap(lambda x: float(x.strip("[]")) if isinstance(x, str) and x.startswith("[") else x)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df["emotion_label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Model Training ===
results = []

# Random Forest
rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_f1 = f1_score(y_test, rf_preds, average="weighted")
results.append(("RandomForest", rf, rf_f1))

# SVM
svm = SVC(C=1, kernel="rbf", probability=True)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_f1 = f1_score(y_test, svm_preds, average="weighted")
results.append(("SVM", svm, svm_f1))

# Neural Network
nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(len(np.unique(y)), activation="softmax")
])
nn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
nn_preds = np.argmax(nn.predict(X_test), axis=1)
nn_f1 = f1_score(y_test, nn_preds, average="weighted")
results.append(("NeuralNet", nn, nn_f1))

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="mlogloss")
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_preds, average="weighted")
results.append(("XGBoost", xgb_model, xgb_f1))

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_f1 = f1_score(y_test, lgb_preds, average="weighted")
results.append(("LightGBM", lgb_model, lgb_f1))

# Voting Ensemble
ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("svm", svm),
        ("xgb", xgb_model),
        ("lgb", lgb_model)
    ],
    voting="soft"
)
ensemble.fit(X_train, y_train)
ensemble_preds = ensemble.predict(X_test)
ensemble_f1 = f1_score(y_test, ensemble_preds, average="weighted")
results.append(("VotingEnsemble", ensemble, ensemble_f1))

# === Save All Models ===
os.makedirs("models", exist_ok=True)
for name, model, score in results:
    filename = f"models/{name}.h5" if name == "NeuralNet" else f"models/{name}.pkl"
    if name == "NeuralNet":
        model.save(filename)
    else:
        joblib.dump(model, filename)
    print(f"✅ Saved {name} with F1={score:.4f} to {filename}")

# === Save Scaler and Label Encoder ===
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("✅ Saved scaler and label encoder to 'models/'")

# === Final Ranking ===
results.sort(key=lambda x: x[2], reverse=True)
print("\n🏁 Final Model Rankings:")
for rank, (name, _, f1) in enumerate(results, 1):
    print(f"{rank}. {name}: F1 Score = {f1:.4f}")
