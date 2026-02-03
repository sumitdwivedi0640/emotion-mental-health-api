import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Cleans and normalizes input text
    """
    text = text.lower()                      # lowercase
    text = re.sub(r"[^a-zA-Z]", " ", text)   # remove punctuation/numbers
    tokens = text.split()                   # tokenize
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens
              if word not in stop_words]    # remove stopwords + lemmatize
    return " ".join(tokens)
