import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

# =========================
# CONFIGURATION
# =========================

API_URL = "https://emotion-mental-health-api.onrender.com/predict"
HISTORY_FILE = "data/history/emotion_history.csv"

st.set_page_config(
    page_title="Emotion-Aware Mental Health Monitoring",
    layout="centered"
)

# =========================
# ENSURE HISTORY FILE EXISTS
# =========================

if not os.path.exists("data/history"):
    os.makedirs("data/history")

if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame(columns=["timestamp", "text", "emotion"])
    df.to_csv(HISTORY_FILE, index=False)

# =========================
# HELPER FUNCTION
# =========================


def load_history():
    return pd.read_csv(HISTORY_FILE)

# =========================
# UI: HEADER
# =========================


st.title("🧠 Emotion-Aware Mental Health Monitoring System")
st.write(
    "This application analyzes your text to detect emotions and "
    "monitors emotional trends over time for mental health awareness."
)

# =========================
# UI: USER INPUT
# =========================

user_text = st.text_area(
    "Enter your thoughts (one sentence per line):",
    height=150,
    placeholder="I feel very tired and hopeless\nI am scared about my future"
)

# =========================
# ANALYSIS BUTTON
# =========================

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        texts = user_text.split("\n")
        payload = {"texts": texts}

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()

                # -------------------------
                # DISPLAY PREDICTIONS
                # -------------------------
                st.subheader("🔍 Predicted Emotions")

                history_df = load_history()
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for text, emotion in zip(
                    result["input_texts"],
                    result["predicted_emotions"]
                ):
                    st.write(f"**Text:** {text}")
                    st.write(f"**Emotion:** {emotion}")
                    st.markdown("---")

                    # Save to history
                    history_df.loc[len(history_df)] = [
                        current_time,
                        text,
                        emotion
                    ]

                history_df.to_csv(HISTORY_FILE, index=False)

                # -------------------------
                # DISPLAY RISK
                # -------------------------
                st.subheader("⚠️ Mental Health Risk Level")
                st.success(result["mental_health_risk"])

            else:
                st.error("Backend error. Please try again later.")

        except Exception as e:
            st.error(f"Connection failed: {e}")

# =========================
# DASHBOARD SECTION
# =========================

st.markdown("---")
st.subheader("📊 Emotion Trend Dashboard")

history_df = load_history()

if history_df.empty:
    st.info("No emotion history available yet. Analyze some text to see trends.")
else:
    # Show raw data
    st.write("📄 Emotion History")
    st.dataframe(history_df)

    # Emotion frequency
    st.subheader("📈 Emotion Frequency")
    emotion_counts = history_df["emotion"].value_counts()
    st.bar_chart(emotion_counts)

    # Time-based trend
    st.subheader("⏳ Emotion Trend Over Time")
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

    emotion_time = (
        history_df
        .groupby([history_df["timestamp"].dt.date, "emotion"])
        .size()
        .unstack(fill_value=0)
    )

    st.line_chart(emotion_time)
