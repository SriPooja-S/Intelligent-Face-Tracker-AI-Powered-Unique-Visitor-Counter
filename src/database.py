"""
database.py
-----------
SQLite persistence layer for the Face Tracker.

Design: images stored on disk, only paths + metadata in database.
  - faces table    : one row per unique registered face
  - events table   : one row per entry or exit event
  - visitor_stats  : single-row running unique visitor count

Added: update_embedding() method to persist the running-mean
embedding update that FaceRegistry performs during re-identification.
This ensures the improved embedding survives across video runs.
"""

import sqlite3
import json
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class Database:
    """SQLite wrapper. WAL mode for resilience against interruptions."""

    def __init__(self, db_path: str = "face_tracker.db"):
        self.db_path = db_path
        self._init_schema()
        logger.info(f"Database initialised at: {db_path}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS faces (
            face_id     TEXT PRIMARY KEY,
            embedding   TEXT NOT NULL,
            first_seen  TEXT NOT NULL,
            crop_path   TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id     TEXT NOT NULL,
            event_type  TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
            timestamp   TEXT NOT NULL,
            image_path  TEXT,
            FOREIGN KEY (face_id) REFERENCES faces(face_id)
        );

        CREATE TABLE IF NOT EXISTS visitor_stats (
            id               INTEGER PRIMARY KEY CHECK(id=1),
            unique_visitors  INTEGER NOT NULL DEFAULT 0,
            last_updated     TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );

        INSERT OR IGNORE INTO visitor_stats (id, unique_visitors) VALUES (1, 0);
        """
        with self._conn() as conn:
            conn.executescript(ddl)

    # ------------------------------------------------------------------ #
    # Face registration                                                    #
    # ------------------------------------------------------------------ #

    def register_face(self, face_id: str, embedding: np.ndarray,
                      crop_path: str = None) -> bool:
        """
        Insert a new unique face. Returns True on success, False if duplicate.
        Also increments the unique visitor counter atomically.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emb_json = json.dumps(embedding.tolist())
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO faces (face_id, embedding, first_seen, crop_path) "
                    "VALUES (?, ?, ?, ?)",
                    (face_id, emb_json, ts, crop_path)
                )
                conn.execute(
                    "UPDATE visitor_stats "
                    "SET unique_visitors=unique_visitors+1, last_updated=? "
                    "WHERE id=1",
                    (ts,)
                )
            logger.info(f"DB: Registered {face_id}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"DB: {face_id} already exists — skipped.")
            return False

    def update_embedding(self, face_id: str, embedding: np.ndarray):
        """
        Persist an updated (running-mean) embedding for an existing face.
        Called after re-identification so the gallery improves over time.
        """
        emb_json = json.dumps(embedding.tolist())
        with self._conn() as conn:
            conn.execute(
                "UPDATE faces SET embedding=? WHERE face_id=?",
                (emb_json, face_id)
            )

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Return all (face_id, embedding) pairs for gallery restoration."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT face_id, embedding FROM faces"
            ).fetchall()
        return [
            (row["face_id"],
             np.array(json.loads(row["embedding"]), dtype=np.float32))
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # Event logging                                                        #
    # ------------------------------------------------------------------ #

    def log_event(self, face_id: str, event_type: str,
                  image_path: str = None) -> int:
        """Log one entry or exit event. Returns new row ID."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO events (face_id, event_type, timestamp, image_path) "
                "VALUES (?, ?, ?, ?)",
                (face_id, event_type, ts, image_path)
            )
        logger.info(f"DB: {event_type.upper()} | {face_id}")
        return cur.lastrowid

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_unique_visitor_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT unique_visitors FROM visitor_stats WHERE id=1"
            ).fetchone()
        return row["unique_visitors"] if row else 0

    def get_events_summary(self, limit: int = 200) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT e.id, e.face_id, e.event_type, e.timestamp, e.image_path,
                       f.first_seen, f.crop_path AS registered_crop
                FROM events e
                LEFT JOIN faces f ON e.face_id = f.face_id
                ORDER BY e.timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_all_faces(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT face_id, first_seen, crop_path "
                "FROM faces ORDER BY first_seen DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            v = conn.execute(
                "SELECT unique_visitors FROM visitor_stats WHERE id=1"
            ).fetchone()["unique_visitors"]
            e = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE event_type='entry'"
            ).fetchone()["c"]
            x = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE event_type='exit'"
            ).fetchone()["c"]
            f = conn.execute("SELECT COUNT(*) AS c FROM faces").fetchone()["c"]
        return {
            "unique_visitors": v,
            "total_entries": e,
            "total_exits": x,
            "registered_faces": f,
        }

    def verify(self):
        """Print a human-readable DB report. Run directly to check data."""
        stats = self.get_stats()
        faces = self.get_all_faces()
        events = self.get_events_summary(limit=10)
        print("\n" + "=" * 55)
        print("DATABASE REPORT")
        print("=" * 55)
        print(f"  Unique visitors  : {stats['unique_visitors']}")
        print(f"  Registered faces : {stats['registered_faces']}")
        print(f"  Entry events     : {stats['total_entries']}")
        print(f"  Exit events      : {stats['total_exits']}")
        print(f"\n  Last 10 events:")
        for e in events:
            img = "✓" if e["image_path"] else "✗"
            print(f"    [{e['event_type'].upper():5}] "
                  f"{e['face_id'][-20:]} | {e['timestamp']} | img:{img}")
        print("=" * 55)