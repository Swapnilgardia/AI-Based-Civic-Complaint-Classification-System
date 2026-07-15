"""
train_image_model.py
---------------------
Trains an image classifier on data/images/<Category>/*.png.

Images are loaded explicitly into numpy arrays and split with sklearn's
train_test_split (stratified), rather than relying on
image_dataset_from_directory's internal splitting, so the train/validation
split is fully transparent and easy to verify.

ARCHITECTURE NOTE: this tries real MobileNetV2 transfer learning first
(weights="imagenet"), which is the architecture proposed in the abstract and
what you should get automatically when running this with normal internet
access. If pretrained weights cannot be downloaded (e.g. this offline
sandbox), it falls back to a compact CNN trained from scratch, which
converges far more reliably than a full MobileNetV2 with random weights on a
small dataset, and prints a clear warning saying so.

Run:
    python src/train_image_model.py
Produces:
    models/image_model.keras
    models/image_label_encoder.joblib
    models/image_metrics.json
"""

import os
import json
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


def load_images_as_arrays():
    class_names = sorted(
        d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))
    )
    X, y = [], []
    for category in class_names:
        cat_dir = os.path.join(IMAGES_DIR, category)
        files = sorted(os.listdir(cat_dir))
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = Image.open(os.path.join(cat_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            X.append(np.asarray(img, dtype=np.float32))
            y.append(category)
    X = np.stack(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images across {len(class_names)} classes: {class_names}")
    return X, y, class_names


def build_transfer_learning_model(num_classes, base_model):
    """Classic MobileNetV2 transfer learning: freeze the pretrained backbone
    and train only a new classification head on top. Used when real ImageNet
    weights were downloaded successfully - this is the architecture proposed
    in the abstract."""
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_fallback_cnn(num_classes):
    """Lightweight CNN trained fully from scratch. Used only when ImageNet
    pretrained weights cannot be downloaded (e.g. this offline sandbox)."""
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(num_classes):
    pretrained = True
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        model = build_transfer_learning_model(num_classes, base_model)
    except Exception as e:
        print("WARNING: could not download ImageNet pretrained weights "
              f"({e}). Falling back to a compact CNN trained from scratch. "
              "When you run this with normal internet access, the script "
              "will automatically use real MobileNetV2 transfer learning "
              "instead, which is the architecture proposed in the abstract.")
        model = build_fallback_cnn(num_classes)
        pretrained = False
    return model, pretrained


def main():
    X, y_text, class_names = load_images_as_arrays()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)
    y = label_encoder.transform(y_text)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print("Train class balance:", np.bincount(y_train))
    print("Val class balance:", np.bincount(y_val))

    model, pretrained = build_model(num_classes=len(class_names))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stop],
        verbose=2,
    )

    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_val, y_pred).tolist()
    val_acc = float(max(history.history["val_accuracy"]))
    print(classification_report(y_val, y_pred, target_names=class_names))

    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "image_label_encoder.joblib"))
    model_path = os.path.join(MODELS_DIR, "image_model.keras")
    model.save(model_path)

    metrics = {
        "pretrained_imagenet_weights_used": pretrained,
        "best_val_accuracy": val_acc,
        "epochs_trained": len(history.history["accuracy"]),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "note": (
            "Trained on procedurally generated synthetic images (see "
            "generate_image_dataset.py) because this environment has no "
            "internet access to a real civic-complaint photo dataset or the "
            "ImageNet weight servers. Replace data/images/<Category>/ with "
            "real photographs and re-run this script for production use; "
            "with internet access it will automatically use real MobileNetV2 "
            "ImageNet transfer learning."
        ),
    }
    with open(os.path.join(MODELS_DIR, "image_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
