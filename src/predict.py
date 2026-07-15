"""
predict.py
----------
Loads the trained text (TF-IDF + Logistic Regression) and image (CNN/
MobileNetV2) models and exposes a single `classify_complaint()` function
that the Flask app calls.

Combination logic:
  - If only text is given -> use the text model's prediction directly.
  - If only an image is given -> use the image model's prediction directly.
  - If both are given -> average the two models' class probability
    distributions (equal weight) and take the argmax. This is a simple,
    explainable multimodal fusion approach appropriate for a course project;
    swap in a learned fusion layer later if you want to improve on it.

If the image model file is missing or fails to load (e.g. TensorFlow not
installed in a lightweight deployment), the app still works using text-only
classification - this keeps the system usable even in constrained
deployments such as small free-tier Render instances.
"""

import os
import numpy as np
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_text_vectorizer = None
_text_model = None
_text_label_encoder = None
_image_model = None
_image_label_encoder = None
_IMG_SIZE = 160


def _load_text_models():
    global _text_vectorizer, _text_model, _text_label_encoder
    if _text_model is None:
        _text_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        _text_model = joblib.load(os.path.join(MODELS_DIR, "text_model.joblib"))
        _text_label_encoder = joblib.load(os.path.join(MODELS_DIR, "text_label_encoder.joblib"))
    return _text_vectorizer, _text_model, _text_label_encoder


def _load_image_models():
    global _image_model, _image_label_encoder
    if _image_model is None:
        try:
            import tensorflow as tf
            model_path = os.path.join(MODELS_DIR, "image_model.keras")
            _image_model = tf.keras.models.load_model(model_path)
            _image_label_encoder = joblib.load(os.path.join(MODELS_DIR, "image_label_encoder.joblib"))
        except Exception as e:
            print(f"Image model unavailable, continuing text-only: {e}")
            _image_model = False  # sentinel meaning "tried and failed"
    return _image_model, _image_label_encoder


def _predict_text_proba(text: str):
    vectorizer, model, label_encoder = _load_text_models()
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    return proba, label_encoder.classes_


def _predict_image_proba(image_path: str):
    from PIL import Image
    image_model, label_encoder = _load_image_models()
    if not image_model:
        return None, None
    img = Image.open(image_path).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, ...]
    proba = image_model.predict(arr, verbose=0)[0]
    return proba, label_encoder.classes_


def classify_complaint(text: str = None, image_path: str = None) -> dict:
    """Returns a dict: {category, confidence, source, details}"""
    text = (text or "").strip()
    has_text = len(text) > 0
    has_image = image_path is not None and os.path.exists(image_path)

    if not has_text and not has_image:
        raise ValueError("At least one of complaint text or an image must be provided.")

    text_proba, text_classes = (None, None)
    image_proba, image_classes = (None, None)

    if has_text:
        text_proba, text_classes = _predict_text_proba(text)
    if has_image:
        image_proba, image_classes = _predict_image_proba(image_path)

    if has_text and image_proba is not None:
        classes = list(text_classes)
        
        aligned_image_proba = np.array([
            image_proba[list(image_classes).index(c)] if c in image_classes else 0.0
            for c in classes
        ])

        text_idx = np.argmax(text_proba)
        image_idx = np.argmax(aligned_image_proba)

        text_conf = text_proba[text_idx]
        image_conf = aligned_image_proba[image_idx]
        
        # If both models agree, average them.
        if text_idx == image_idx:
            combined = (text_proba + aligned_image_proba) / 2

        # If they disagree, trust the more confident model.
        elif text_conf >= image_conf:
          combined = text_proba
        else:
          combined = aligned_image_proba

        idx = int(np.argmax(combined))
        category = classes[idx]
        confidence = float(combined[idx])
        source = "text+image (fused)"

        
    elif has_text:
        idx = int(np.argmax(text_proba))
        category = text_classes[idx]
        confidence = float(text_proba[idx])
        source = "text-only"
    else:
        idx = int(np.argmax(image_proba))
        category = image_classes[idx]
        confidence = float(image_proba[idx])
        source = "image-only"

    return {
        "category": str(category),
        "confidence": confidence,
        "source": source,
    }
