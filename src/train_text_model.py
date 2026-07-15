"""
train_text_model.py
--------------------
Trains a TF-IDF + Logistic Regression classifier on data/complaints_text.csv.
Performs a real train/test split, cross-validation, and saves:
    models/tfidf_vectorizer.joblib
    models/text_model.joblib
    models/text_label_encoder.joblib
    models/text_metrics.json   (accuracy, classification report, CV scores)

Run:
    python src/train_text_model.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "complaints_text_cleaned.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, categories: {df['category'].unique().tolist()}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    X_text = df["complaint_text"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=2,random_state=42)
    model.fit(X_train_vec, y_train)

    # Real 5-fold cross validation on the training split
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Test accuracy: {acc:.4f}")
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(model, os.path.join(MODELS_DIR, "text_model.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "text_label_encoder.joblib"))

    metrics = {
        "test_accuracy": acc,
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": label_encoder.classes_.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(os.path.join(MODELS_DIR, "text_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
