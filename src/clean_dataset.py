"""
clean_dataset.py
----------------
Cleans complaint text before training.

Fixes:
- Out_of_sewer_vent_4_inch
        ->
  Out of sewer vent 4 inch

- Outofsewervent4inch
        ->
  Out of sewer vent 4 inch

- CamelCase words
- Numbers attached to words
- Multiple spaces

Creates:
    data/complaints_text_cleaned.csv

Run:
    python src/clean_dataset.py
"""

import os
import re
import pandas as pd
import wordninja

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_FILE = os.path.join(BASE_DIR, "data", "complaints_text.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "complaints_text_cleaned.csv")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Replace underscores with spaces
    text = text.replace("_", " ")

    # Split CamelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Separate letters and numbers
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

    words = []

    for token in text.split():

        # Skip numbers
        if token.isdigit():
            words.append(token)
            continue

        # Already normal word
        if len(token) <= 2:
            words.append(token)
            continue

        # Use wordninja only for long joined words
        split_words = wordninja.split(token)

        if len(split_words) > 1:
            words.extend(split_words)
        else:
            words.append(token)

    text = " ".join(words)

    # Remove duplicate spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} complaints")

    df["complaint_text"] = df["complaint_text"].apply(clean_text)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned dataset to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()