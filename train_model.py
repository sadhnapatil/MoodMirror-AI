import os
import librosa
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

dataset_path = "dataset"

features = []
labels = []

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):

            emotion = file.split("-")[2]

            if emotion not in emotion_map:
                continue

            label = emotion_map[emotion]

            file_path = os.path.join(root, file)

            audio, sr = librosa.load(file_path)

            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=40
            )

            feature = np.mean(mfccs.T, axis=0)

            features.append(feature)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300
)

model.fit(X_train, y_train)

print("Accuracy:",
      model.score(X_test, y_test))

joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model Saved")