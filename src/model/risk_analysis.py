import pandas as pd
from collections import Counter

# Emotions considered negative
NEGATIVE_EMOTIONS = {"sadness", "fear", "anger"}


def assess_risk(emotion_list):
    """
    Takes a list of predicted emotions and returns risk level
    """
    total = len(emotion_list)
    counts = Counter(emotion_list)

    negative_count = sum(counts[e] for e in NEGATIVE_EMOTIONS if e in counts)

    ratio = negative_count / total

    if ratio >= 0.6:
        return "HIGH RISK"
    elif ratio >= 0.3:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


# Example usage
if __name__ == "__main__":
    sample_emotions = [
        "sadness", "fear", "sadness",
        "anger", "sadness", "fear"
    ]
    print("Risk Level:", assess_risk(sample_emotions))
