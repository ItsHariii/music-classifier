# Music Emotion Classifier 🎵

This project is an AI-powered tool that classifies the emotion of a music/audio file into one of four categories: **happy**, **sad**, **angry**, or **calm**. It supports `.wav`, `.mp3`, and `.mid` files.

## Features

- Feature extraction using Librosa (MFCCs, delta MFCCs, chroma, spectral contrast, rolloff, bandwidth, tonnetz, tempo, etc.)
- Trains multiple ML models: Random Forest, SVM, Neural Network, XGBoost, LightGBM
- Voting Ensemble for best performance
- Web-based frontend using Streamlit
- Supports real-time emotion prediction for audio uploads

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/music-emotion-classifier.git
cd music-emotion-classifier
```

### 2. Install Dependencies

Ensure Python 3.9+ is installed, then run:

```bash
pip install -r requirements.txt
```

If `librosa` fails due to `ffmpeg`, try:

```bash
brew install ffmpeg
```

---

## Dataset

This project uses:

- **MEC** (Musical Emotion Classification)
- **DEAM** (Dynamic Emotion Annotations in Music)
- **EMOPIA** (Emotional Pop Piano dataset)

Due to file size limits, the audio files are not included. You can download them here:

- MEC: https://www.kaggle.com/datasets/kingofarmy/musical-emotions-classification?resource=download
- DEAM: https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music?utm_source=chatgpt.com
- EMOPIA: https://zenodo.org/records/5257995

Place them inside the `data/` directory as follows:

```
data/
├── mec/
│   └── audio/
│       ├── train/Happy
│       └── test/Sad
├── deam/
│   ├── audio_wav/
│   └── static_annotations_*.csv
└── emopia/
    ├── audio_wav/
    └── metadata_by_song.csv
```

---

## Run the Pipeline

### 1. Extract Features

```bash
python src/audio_features_extractor.py
```

This will process MEC, DEAM, and EMOPIA and save features to CSVs.

### 2. Train Models

```bash
python src/train_models.py
```

This trains and saves all models to the `models/` directory.

### 3. Predict from Audio File

```bash
python src/predict.py path/to/audio.mp3
```

This will load the voting ensemble and output the predicted emotion.

---

## Launch the Streamlit Web App

```bash
streamlit run app.py
```

Upload a `.wav`, `.mp3`, or `.mid` file and the model will display the predicted emotion.

---

## Project Structure

```
.
├── app.py                     # Streamlit frontend
├── src/
│   ├── audio_features_extractor.py
│   ├── train_models.py
│   └── predict.py
├── models/                    # Saved models, scaler, label encoder
├── data/                      # Place datasets here
├── requirements.txt
└── README.md
```

---

## Author

Hari Thavittupalayam Manivannan
Dhanush Reddy Pucha

