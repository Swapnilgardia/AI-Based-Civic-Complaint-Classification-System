"""
db.py
-----
Minimal SQLite data-access layer for storing civic complaints.

Table: complaints
    id              TEXT PRIMARY KEY   (unique complaint ID, e.g. CMP-8F3A2C1B)
    complaint_text  TEXT
    image_filename  TEXT (nullable)
    category        TEXT
    confidence      REAL
    source          TEXT   ("text-only" / "image-only" / "text+image (fused)")
    priority        TEXT
    flagged_for_review INTEGER (0/1)
    status          TEXT   ("Open" / "In Progress" / "Resolved")
    created_at      TEXT   (ISO timestamp)
"""

import os
import sqlite3
import uuid
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "database", "complaints.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id TEXT PRIMARY KEY,
            complaint_text TEXT,
            image_filename TEXT,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            priority TEXT NOT NULL,
            flagged_for_review INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'Open',
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def generate_complaint_id() -> str:
    return "CMP-" + uuid.uuid4().hex[:8].upper()


def insert_complaint(complaint_text, image_filename, category, confidence,
                      source, priority, flagged_for_review):
    conn = get_connection()
    complaint_id = generate_complaint_id()
    conn.execute(
        """INSERT INTO complaints
           (id, complaint_text, image_filename, category, confidence, source,
            priority, flagged_for_review, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            complaint_id, complaint_text, image_filename, category, confidence,
            source, priority, int(flagged_for_review), "Open",
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ),
    )
    conn.commit()
    conn.close()
    return complaint_id


def get_complaint(complaint_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_complaints(category=None, priority=None, status=None, limit=200):
    conn = get_connection()
    query = "SELECT * FROM complaints WHERE 1=1"
    params = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_status(complaint_id, status):
    conn = get_connection()
    conn.execute("UPDATE complaints SET status = ? WHERE id = ?", (status, complaint_id))
    conn.commit()
    conn.close()

def delete_complaint(complaint_id):
    conn = get_connection()
    conn.execute(
        "DELETE FROM complaints WHERE id = ?",
        (complaint_id,)
    )
    conn.commit()
    conn.close()

def get_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) c FROM complaints").fetchone()["c"]
    by_category = conn.execute(
        "SELECT category, COUNT(*) c FROM complaints GROUP BY category"
    ).fetchall()
    by_priority = conn.execute(
        "SELECT priority, COUNT(*) c FROM complaints GROUP BY priority"
    ).fetchall()
    by_status = conn.execute(
        "SELECT status, COUNT(*) c FROM complaints GROUP BY status"
    ).fetchall()
    conn.close()
    return {
        "total": total,
        "by_category": {r["category"]: r["c"] for r in by_category},
        "by_priority": {r["priority"]: r["c"] for r in by_priority},
        "by_status": {r["status"]: r["c"] for r in by_status},
    }
