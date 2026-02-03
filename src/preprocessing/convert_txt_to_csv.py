import pandas as pd


def convert_txt_to_csv(txt_path, csv_path):
    texts = []
    emotions = []

    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if ';' in line:
                text, emotion = line.split(';')
                texts.append(text)
                emotions.append(emotion)

    df = pd.DataFrame({
        'text': texts,
        'emotion': emotions
    })

    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


# Convert all files
convert_txt_to_csv("data/raw/train.txt", "data/processed/train.csv")
convert_txt_to_csv("data/raw/val.txt", "data/processed/val.csv")
convert_txt_to_csv("data/raw/test.txt", "data/processed/test.csv")
