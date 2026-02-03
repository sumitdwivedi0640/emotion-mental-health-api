import pandas as pd
from src.preprocessing.text_cleaning import clean_text

# Load processed CSVs
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Apply cleaning
train_df["clean_text"] = train_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

# Save cleaned files
train_df.to_csv("data/processed/train_clean.csv", index=False)
val_df.to_csv("data/processed/val_clean.csv", index=False)
test_df.to_csv("data/processed/test_clean.csv", index=False)

print("Cleaned CSV files created successfully.")
