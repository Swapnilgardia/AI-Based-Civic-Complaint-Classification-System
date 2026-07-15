# AI-Based Civic Complaint Classification System

A full-stack ML system that classifies citizen civic complaints (Garbage, Road
Damage, Water Leakage, Drainage Blockage, Streetlight Fault) from text
descriptions and/or photos, assigns a priority level, and stores records in a
database — built from the project abstract "AI-Based Civic Complaint
Classification System Using NLP and Deep Learning."

## 🌐 Live Demo

**Project Website:** https://civic-complaint-classifier.onrender.com/

## What's actually inside

This is a genuinely trained, end-to-end system, not an API wrapper:

| Component | Technique | Real training performed |
|---|---|---|
| Text classifier | TF-IDF (1-2 grams) + Logistic Regression | Yes — 45,000 samples, 80/20 split + 5-fold CV |
| Image classifier | CNN (MobileNetV2, ImageNet transfer learning) | Yes — 1,500 images, 80/20 split |
| Priority assignment | Rule-based, per abstract's requirement | N/A (rules, not ML, by design) |
| Storage | SQLite | N/A |

**Dataset note:**

Both the text and image datasets used to train these models were sourced
from Kaggle, not synthetically or procedurally generated. `data/complaints_text.csv`
contains 45,000 labeled complaint text samples, and `data/images/<Category>/`
contains 1,500 labeled civic issue photographs (300 per class). The image
model was trained using genuine MobileNetV2 transfer learning with pretrained
ImageNet weights, exactly as proposed in the abstract.

## Architecture

```
Citizen submits complaint (text and/or photo)
        │
        ▼
 ┌─────────────────┐     ┌──────────────────────┐
 │ TF-IDF + LogReg  │     │ MobileNetV2           │
 │ (text_model)     │     │ (image_model)         │
 └────────┬─────────┘     └──────────┬───────────┘
          │  category probabilities   │
          └───────────┬───────────────┘
                       ▼
            Fuse (average) if both given
                       ▼
              Final category + confidence
                       ▼
            Rule-based priority assignment
                       ▼
         Store in SQLite + generate complaint ID
                       ▼
              Flask UI shows result / dashboard
```

## Project structure

```
civic_complaint_classifier/
├── app.py                       # Flask application (routes)
├── requirements.txt             # Full deps (incl. TensorFlow for image model)
├── requirements-light.txt       # Text-only deps (no TensorFlow, for tight hosts)
├── Procfile                     # gunicorn start command (Render/Heroku style)
├── render.yaml                  # Render.com deployment blueprint
├── data/
│   ├── complaints_text.csv      # Kaggle text dataset (45,000 rows)
│   └── images/<Category>/       # Kaggle image dataset (1,500 images)
├── models/                      # Trained model artifacts (see below)
├── src/
│   ├── train_text_model.py
│   ├── train_image_model.py
│   ├── predict.py               # Loads models, fuses predictions
│   └── priority.py              # Rule-based priority logic
├── database/
│   └── db.py                    # SQLite data access layer
├── templates/                   # Jinja2 HTML templates
└── static/
    ├── css/style.css
    └── uploads/                 # User-uploaded complaint photos
```

### Trained model artifacts in `models/`

| File | What it is |
|---|---|
| `tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer |
| `text_model.joblib` | Trained Logistic Regression classifier |
| `text_label_encoder.joblib` | Category label encoder for the text model |
| `text_metrics.json` | Real accuracy, CV scores, classification report, confusion matrix |
| `image_model.keras` | Trained MobileNetV2 model |
| `image_label_encoder.joblib` | Category label encoder for the image model |
| `image_metrics.json` | Real accuracy, training history, classification report, confusion matrix |

## Final trained model metrics

### 1. Text Classification Model (TF-IDF + Logistic Regression)

**Dataset**

* Total samples: **45,000**
* Train/Test Split: **80% / 20%**
* Training samples: **36,000**
* Test samples: **9,000**

**Metrics**

| Metric | Value |
| --- | ---: |
| Test Accuracy | **98.44%** |
| Cross Validation Accuracy | **98.26%** |
| Cross Validation Std. Dev. | **±0.14%** |

**Per-Class Performance**

| Class | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| Drainage Blockage | **1.00** | **0.99** | **0.99** |
| Garbage | **1.00** | **0.96** | **0.98** |
| Road Damage | **1.00** | **0.98** | **0.99** |
| Streetlight Fault | **1.00** | **0.99** | **0.99** |
| Water Leakage | **0.93** | **1.00** | **0.96** |

**Overall**

| Metric | Value |
| --- | ---: |
| Accuracy | **98.44%** |
| Macro Precision | **0.99** |
| Macro Recall | **0.98** |
| Macro F1-Score | **0.98** |
| Weighted F1-Score | **0.98** |

---

### 2. Image Classification Model (MobileNetV2)

**Dataset**

* Total images: **1,500**
* Classes: **5**
* Images per class: **300**
* Training images: **1,200**
* Validation images: **300**
* Train/Validation Split: **80% / 20%**

**Model**

* **MobileNetV2 Transfer Learning**
* **ImageNet Pretrained Weights**
* Image Size: **160 × 160**
* Batch Size: **32**
* Epochs Trained: **14** (EarlyStopping)

**Metrics**

| Metric | Value |
| --- | ---: |
| Training Accuracy | **98.00%** |
| Validation Accuracy | **97.67%** |
| Validation Loss | **0.0806** |

**Per-Class Performance**

| Class | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| Drainage Blockage | **0.95** | **0.97** | **0.96** |
| Garbage | **0.98** | **1.00** | **0.99** |
| Road Damage | **0.97** | **0.95** | **0.96** |
| Streetlight Fault | **0.98** | **0.97** | **0.97** |
| Water Leakage | **1.00** | **1.00** | **1.00** |

**Overall**

| Metric | Value |
| --- | ---: |
| Accuracy | **97.67%** |
| Macro Precision | **0.98** |
| Macro Recall | **0.98** |
| Macro F1-Score | **0.98** |
| Weighted F1-Score | **0.98** |

---

### Final Project Performance Summary

| Model | Algorithm | Accuracy |
| --- | --- | ---: |
| **Text Complaint Classifier** | TF-IDF + Logistic Regression | **98.44%** |
| **Image Complaint Classifier** | MobileNetV2 (Transfer Learning) | **97.67%** |

## Running locally

```bash
cd civic_complaint_classifier
pip install -r requirements.txt          # or requirements-light.txt for text-only
python app.py                            # runs on http://localhost:5000
```

Open http://localhost:5000, submit a complaint with text and/or a photo, and
you'll be redirected to a result page with your complaint ID, predicted
category, priority, and confidence. Visit `/dashboard` to see all complaints,
filter by category/priority/status, and update status.

## Retraining the models

```bash
python src/train_text_model.py
python src/train_image_model.py
```

Both scripts overwrite the artifacts in `models/` and print/save real
metrics — no manual metric entry anywhere.

## Deploying to Render

1. Push this project to a GitHub repository.
2. On Render.com: **New +** → **Blueprint** → connect your repo (Render will
   detect `render.yaml`), or **New +** → **Web Service** manually with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
3. **Memory note:** `tensorflow-cpu` is a large dependency. Render's free
   instance (512 MB RAM) may struggle to load both TensorFlow and the model.
   If the deploy fails or the service crashes/OOMs:
   - Use `requirements-light.txt` instead (rename to `requirements.txt` or
     point Render's build at it) and remove `models/image_model.keras` — the
     app automatically detects the missing image model and falls back to
     **text-only classification** without any code change.
   - Or upgrade to a paid Render instance with more RAM.
4. **Persistence note:** SQLite (`database/complaints.db`) is a file on disk.
   Render's free web services use an **ephemeral filesystem** — data is lost
   on restart/redeploy unless you attach a paid persistent disk (see the
   `disk:` block in `render.yaml`, which requires a paid plan). For a purely
   free-tier demo, treat complaint history as session-only, or swap in a free
   external database (e.g. Render's free PostgreSQL) for real persistence.

## Priority rules (editable in `src/priority.py`)

| Category | Priority | Rationale |
|---|---|---|
| Water Leakage | High | Wastage of potable water, possible flooding |
| Road Damage | High | Accident risk |
| Drainage Blockage | Medium | Health hazard, worsens in monsoon |
| Garbage | Medium | Health/hygiene hazard |
| Streetlight Fault | Low | Safety concern, not immediately hazardous |

Complaints classified with confidence below 55% are flagged for manual
review regardless of category, since low-confidence auto-classification may
be unreliable.

## Tech stack

Python, Flask, gunicorn, scikit-learn, TensorFlow/Keras, Pandas, NumPy,
Pillow, joblib, SQLite, HTML/CSS (Jinja2 templates).

