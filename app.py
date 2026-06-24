import matplotlib.pyplot as plt

import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from fpdf import FPDF
from deep_translator import GoogleTranslator
from datetime import datetime
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="EmotionMirror AI",
    
    page_icon="🎙️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# ---------------- HISTORY FILE ----------------

if not os.path.exists("emotion_history.csv"):
    pd.DataFrame(
        columns=["Time", "Emotion"]
    ).to_csv(
        "emotion_history.csv",
        index=False
    )

# ---------------- HEADER ----------------

st.title("🎙️ EmotionMirror AI")
st.markdown("""
### 🧠 AI Powered Speech Emotion Analysis

Detect emotions from voice recordings,
analyze emotional patterns,
track mood trends,
and receive wellness recommendations.
""")
st.subheader(
    "Real-Time Speech Emotion Recognition & Wellness Assistant"
)

language = st.selectbox(
    "🌍 Select Language",
    ["English", "Hindi", "Marathi"]
)

st.subheader("🎤 Record or Upload Audio")

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)


# ---------------- MAIN APP ----------------
if uploaded_file:

    # Load audio
    audio, sr = librosa.load(uploaded_file)

    # Audio player
    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    st.success("✅ Audio uploaded successfully")

    # Waveform
    st.subheader("📈 Voice Waveform")

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(audio)
    ax.set_title("Voice Signal")

    st.pyplot(fig)

    # Spectrogram
    st.subheader("🎼 Voice Spectrogram")

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    fig2, ax2 = plt.subplots(figsize=(10,4))

    ax2.imshow(
        D,
        aspect="auto",
        origin="lower"
    )

    ax2.set_title("Spectrogram")

    st.pyplot(fig2)

    # MFCC Feature Extraction
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    feature = np.mean(
        mfccs.T,
        axis=0
    )

    feature = scaler.transform([feature])

    prediction = model.predict(feature)

    emotion = encoder.inverse_transform(
        prediction
    )[0]




    

    # Emotion Emoji

    emoji_map = {
        "happy":"😄",
        "sad":"😢",
        "angry":"😠",
        "fearful":"😨",
        "calm":"😌",
        "neutral":"🙂"
    }

    st.markdown(
        f"# {emoji_map.get(emotion,'🙂')}"
    )

    st.success(
        f"🎭 Detected Emotion: {emotion.upper()}"
    )

    # Mood Score

    scores = {
        "happy":100,
        "calm":85,
        "neutral":70,
        "fearful":40,
        "sad":30,
        "angry":20
    }

    mood_score = scores.get(
        emotion,
        50
    )

    st.metric(
        "💚 Mood Score",
        f"{mood_score}/100"
    )

    # Save History

    pd.DataFrame({
        "Time":[datetime.now()],
        "Emotion":[emotion]
    }).to_csv(
        "emotion_history.csv",
        mode="a",
        header=False,
        index=False
    )

    # Voice Energy

    energy = np.mean(audio ** 2)

    energy_percent = min(
        int(energy * 10000),
        100
    )

    st.metric(
        "🔊 Voice Energy",
        f"{energy_percent}%"
    )

    # Stress Level

    if energy_percent > 70:
        stress = "High"

    elif energy_percent > 40:
        stress = "Medium"

    else:
        stress = "Low"

    st.warning(
        f"⚠️ Stress Level: {stress}"
    )

    # Real Confidence Score

    probabilities = model.predict_proba(
        feature
    )[0]

    confidence = round(
        np.max(probabilities) * 100,
        2
    )

    st.metric(
        "🎯 Prediction Confidence",
        f"{confidence}%"
    )

    # Wellness Coach

    suggestions = {

        "happy":
        "Keep spreading positivity 😊",

        "sad":
        "Take a short walk and talk with a friend ❤️",

        "angry":
        "Try deep breathing exercises 🌿",

        "fearful":
        "Focus on one task at a time 🌟",

        "calm":
        "Great emotional balance 😌",

        "neutral":
        "Maintain healthy routines 👍"
    }

    message = suggestions.get(
        emotion,
        "Stay positive"
    )

    if language == "Hindi":

        message = GoogleTranslator(
            source="auto",
            target="hi"
        ).translate(message)

    elif language == "Marathi":

        message = GoogleTranslator(
            source="auto",
            target="mr"
        ).translate(message)

    st.info(
        f"🤖 AI Wellness Coach\n\n{message}"
    )

    # Current Emotion Chart

    st.subheader(
        "📊 Current Emotion"
    )

    df = pd.DataFrame({
        "Emotion":[emotion]
    })

    fig = px.histogram(
        df,
        x="Emotion",
        title="Detected Emotion"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # History

    history = pd.read_csv(
    "emotion_history.csv",
    header=None,
    names=["Timestamp", "Emotion"]
)

    st.subheader(
        "📈 Emotion Timeline"
    )

    st.dataframe(
        history.tail(10)
    )

    # Pie Chart

    st.subheader(
        "🥧 Weekly Emotion Analytics"
    )

    fig_pie = px.pie(
        history,
        names="Emotion",
        title="Emotion Distribution"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # Bar Chart

    if len(history) > 1:

        emotion_count = (
            history["Emotion"]
            .value_counts()
            .reset_index()
        )

        emotion_count.columns = [
            "Emotion",
            "Count"
        ]

        fig2 = px.bar(
            emotion_count,
            x="Emotion",
            y="Count",
            title="Emotion History"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        dominant = (
            history["Emotion"]
            .value_counts()
            .idxmax()
        )

        st.info(
            f"""
📊 Emotional Insight

Most frequent emotion:
{dominant}

Continue monitoring
your emotional trends.
"""
        )

    # PDF Report

    if st.button(
        "📄 Generate PDF Report"
    ):

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font(
            "Arial",
            size=12
        )

        pdf.cell(
            200,
            10,
            txt="EmotionMirror AI Report",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Emotion: {emotion}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Stress Level: {stress}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Voice Energy: {energy_percent}%",
            ln=True
        )

        pdf.output(
            "emotion_report.pdf"
        )

        with open(
    "emotion_report.pdf",
    "rb"
) as file:
         st.download_button(
        "⬇ Download Report",
        file,
        "Emotion_Report.pdf"
    )
