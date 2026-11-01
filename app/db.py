"""SQLite prediction log.

Every prediction the API serves is stored, so we can audit decisions,
compute business stats with SQL, and feed the drift monitor.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.features import ROOT

DB_PATH = Path(os.getenv("CHURN_DB_PATH", ROOT / "data" / "predictions.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    features     TEXT    NOT NULL,   -- JSON of the request
    churn_proba  REAL    NOT NULL,
    risk_level   TEXT    NOT NULL,
    model_name   TEXT    NOT NULL
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def log_predictions(rows: list[tuple[dict, float, str]], model_name: str) -> None:
    with connect() as conn:
        # Parameterised query (the ? placeholders) prevents SQL injection.
        conn.executemany(
            "INSERT INTO predictions (features, churn_proba, risk_level, model_name) "
            "VALUES (?, ?, ?, ?)",
            [(json.dumps(f), p, r, model_name) for f, p, r in rows],
        )


def recent_features(limit: int = 1000) -> list[dict]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT features FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        return [json.loads(r["features"]) for r in cur.fetchall()]


def summary_stats() -> dict:
    with connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n, AVG(churn_proba) AS avg_p FROM predictions"
        ).fetchone()
        by_risk = conn.execute(
            "SELECT risk_level, COUNT(*) AS n FROM predictions GROUP BY risk_level"
        ).fetchall()
    return {
        "total_predictions": total["n"],
        "avg_churn_probability": round(total["avg_p"] or 0.0, 4),
        "by_risk_level": {r["risk_level"]: r["n"] for r in by_risk},
    }
