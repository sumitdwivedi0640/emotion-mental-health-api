import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load test data
test_df = pd.read_csv("data/processed/test_clean.csv")

X_test = test_df["clean_text"]
y_test = test_df["emotion"]

# Load saved model artifacts
model = pickle.load(open("models/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

# Encode labels
y_test_enc = label_encoder.transform(y_test)

# Vectorize text
X_test_vec = vectorizer.transform(X_test)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test_enc, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test_enc, y_pred,
      target_names=label_encoder.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test_enc, y_pred))
