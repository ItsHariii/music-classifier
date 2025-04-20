import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# === Load Processed Data ===
df = pd.read_csv("/Users/harimanivannan/Documents/GitHub/music-classifier/data/processed_dataset.csv")
features = [col for col in df.columns if col not in ["emotion", "source"]]

X = df[features].values
y = df["emotion"].values

# === Encode Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Class Weights (for imbalance) ===
classes = np.unique(y_train)
class_weights_dict = dict(zip(classes, compute_class_weight(class_weight="balanced", classes=classes, y=y_train)))

# === RANDOM FOREST ===
print("\n--- Random Forest (Tuned) ---")
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None]
}
grid_rf = GridSearchCV(rf, rf_grid, cv=3)
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.predict(X_test)

print("Best Params:", grid_rf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf, average="weighted"))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# === SVM ===
print("\n--- Support Vector Machine (Tuned) ---")
svm = SVC(class_weight="balanced", probability=True)
svm_grid = {
    'C': [0.5, 1, 10],
    'kernel': ['rbf', 'poly']
}
grid_svm = GridSearchCV(svm, svm_grid, cv=3)
grid_svm.fit(X_train, y_train)
y_pred_svm = grid_svm.predict(X_test)

print("Best Params:", grid_svm.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm, average="weighted"))
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# === NEURAL NETWORK ===
print("\n--- Neural Network (Improved) ---")
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate NN
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
y_pred_nn = model.predict(X_test)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_nn_classes))
print("F1 Score:", f1_score(y_test, y_pred_nn_classes, average="weighted"))
print(classification_report(y_test, y_pred_nn_classes, target_names=le.classes_))
