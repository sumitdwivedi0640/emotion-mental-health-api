from flask import Flask, request, jsonify
import pickle
import os

from src.preprocessing.text_cleaning import clean_text
from src.model.risk_analysis import assess_risk

# Load model artifacts
model = pickle.load(open("models/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return {"message": "Emotion-Aware Mental Health Monitoring API is running"}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "texts" not in data:
        return jsonify({"error": "Please provide a list of texts"}), 400

    texts = data["texts"]

    # Clean text
    cleaned_texts = [clean_text(text) for text in texts]

    # Vectorize
    vectors = vectorizer.transform(cleaned_texts)

    # Predict emotions
    preds = model.predict(vectors)
    emotions = label_encoder.inverse_transform(preds)

    # Risk analysis
    risk = assess_risk(emotions)

    return jsonify({
        "input_texts": texts,
        "predicted_emotions": emotions.tolist(),
        "mental_health_risk": risk
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
