"""SQLite com consultas parametrizadas; nenhuma SQL gerada pela LLM."""

import json
import sqlite3
from pathlib import Path


class Repository:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS patients(id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    event TEXT NOT NULL, details TEXT NOT NULL,
                    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS decisions(
                    run_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL, approved INTEGER NOT NULL,
                    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path)

    def seed(self, path: Path):
        patients = json.loads(path.read_text(encoding="utf-8"))
        ids = [p["id"] for p in patients]
        if len(ids) != len(set(ids)) or any(p.get("synthetic") is not True for p in patients):
            raise ValueError("Somente fixtures sintéticas com IDs únicos são aceitas.")
        with self.connect() as db:
            for p in patients:
                db.execute("INSERT OR IGNORE INTO patients VALUES (?, ?)",
                           (p["id"], json.dumps(p, ensure_ascii=False)))

    def patient(self, patient_id: str):
        with self.connect() as db:
            row = db.execute("SELECT body FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def log(self, run_id: str, event: str, details: dict):
        with self.connect() as db:
            db.execute("INSERT INTO audit(run_id,event,details) VALUES (?,?,?)",
                       (run_id, event, json.dumps(details, ensure_ascii=False)))

    def decision(self, run_id: str, patient_id: str, reviewer: str, approved: bool):
        with self.connect() as db:
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                       (run_id, patient_id, reviewer, int(approved)))

    def events(self, run_id: str):
        with self.connect() as db:
            rows = db.execute(
                "SELECT event,details FROM audit WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()
        return [{"event": name, "details": json.loads(details)} for name, details in rows]
