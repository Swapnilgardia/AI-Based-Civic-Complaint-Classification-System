"""
app.py
------
Flask backend for the AI-Based Civic Complaint Classification System.

Routes:
    GET  /                  -> complaint submission form
    POST /submit            -> classify + store a new complaint, redirect to result
    GET  /result/<id>       -> show classification result + complaint ID for one complaint
    GET  /dashboard         -> list/filter all complaints, view stats
    POST /update_status/<id>-> mark a complaint's status (Open/In Progress/Resolved)
    GET  /health            -> simple health check (used by Render)
"""

import os
import sys
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "database"))

from predict import classify_complaint  # noqa: E402
from priority import assign_priority  # noqa: E402
import db  # noqa: E402

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

os.makedirs(UPLOAD_DIR, exist_ok=True)
db.init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    complaint_text = request.form.get("complaint_text", "").strip()
    image_file = request.files.get("complaint_image")

    image_path = None
    saved_filename = None
    if image_file and image_file.filename and allowed_file(image_file.filename):
        ext = image_file.filename.rsplit(".", 1)[1].lower()
        saved_filename = f"{uuid.uuid4().hex}.{ext}"
        image_path = os.path.join(UPLOAD_DIR, saved_filename)
        image_file.save(image_path)

    if not complaint_text and not image_path:
        flash("Please enter a complaint description and/or upload an image.", "error")
        return redirect(url_for("index"))

    try:
        result = classify_complaint(text=complaint_text, image_path=image_path)
    except Exception as e:
        flash(f"Could not classify complaint: {e}", "error")
        return redirect(url_for("index"))

    priority_info = assign_priority(result["category"], result["confidence"])

    complaint_id = db.insert_complaint(
        complaint_text=complaint_text,
        image_filename=saved_filename,
        category=result["category"],
        confidence=result["confidence"],
        source=result["source"],
        priority=priority_info["priority"],
        flagged_for_review=priority_info["flagged_for_review"],
    )

    return redirect(url_for("result", complaint_id=complaint_id))


@app.route("/result/<complaint_id>")
def result(complaint_id):
    complaint = db.get_complaint(complaint_id)
    if not complaint:
        flash("Complaint not found.", "error")
        return redirect(url_for("index"))
    return render_template("result.html", complaint=complaint)


@app.route("/dashboard")
def dashboard():
    category = request.args.get("category") or None
    priority = request.args.get("priority") or None
    status = request.args.get("status") or None
    complaints = db.list_complaints(category=category, priority=priority, status=status)
    complaints = [
        c for c in complaints
        if c["status"] != "Resolved"
        ]
    stats = db.get_stats()
    categories = ["Garbage", "Road Damage", "Water Leakage", "Drainage Blockage", "Streetlight Fault"]
    priorities = ["High", "Medium", "Low"]
    statuses = ["Open", "In Progress"]
    return render_template(
        "dashboard.html",
        complaints=complaints,
        stats=stats,
        categories=categories,
        priorities=priorities,
        statuses=statuses,
        selected_category=category,
        selected_priority=priority,
        selected_status=status,
    )

@app.route("/resolved")
def resolved():
    complaints = db.list_complaints(status="Resolved")

    return render_template(
        "resolved.html",
        complaints=complaints
    )

@app.route("/update_status/<complaint_id>", methods=["POST"])
def update_status(complaint_id):
    new_status = request.form.get("status")
    if new_status in ("Open", "In Progress", "Resolved"):
        db.update_status(complaint_id, new_status)
        flash(f"Complaint {complaint_id} marked as {new_status}.", "success")
    return redirect(url_for("dashboard"))


@app.route("/delete/<complaint_id>", methods=["POST"])
def delete_complaint(complaint_id):
    db.delete_complaint(complaint_id)
    flash(f"Complaint {complaint_id} deleted successfully.", "success")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
