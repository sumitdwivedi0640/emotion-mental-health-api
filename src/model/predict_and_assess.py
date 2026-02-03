import pickle
from src.preprocessing.text_cleaning import clean_text
from src.model.risk_analysis import assess_risk

# Load model artifacts
model = pickle.load(open("models/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))


def predict_emotions(text_list):
    cleaned = [clean_text(t) for t in text_list]
    vectors = vectorizer.transform(cleaned)
    preds = model.predict(vectors)
    return label_encoder.inverse_transform(preds)


if __name__ == "__main__":
    texts = [
        "I feel very tired and hopeless",
        "Nothing makes me happy anymore",
        "I am scared about my future",
        "I feel angry at myself"
    ]

    emotions = predict_emotions(texts)
    print("Predicted emotions:", emotions)

    risk = assess_risk(emotions)
    print("Mental Health Risk:", risk)
