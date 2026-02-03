import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load cleaned data
train_df = pd.read_csv("data/processed/train_clean.csv")
val_df = pd.read_csv("data/processed/val_clean.csv")

# Features & labels
X_train = train_df["clean_text"]
y_train = train_df["emotion"]

X_val = val_df["clean_text"]
y_val = val_df["emotion"]

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train_enc)

# Evaluation
y_pred = model.predict(X_val_vec)
print(classification_report(y_val_enc, y_pred,
      target_names=label_encoder.classes_))

# Save model artifacts
pickle.dump(model, open("models/emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("models/label_encoder.pkl", "wb"))
