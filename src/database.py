"""
database.py
-----------
SQLite persistence layer for the Face Tracker application.
Handles metadata storage, dynamic dwell-time calculation, and system recovery.
"""

import sqlite3
import json
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class Database:
    """
    SQLite wrapper utilizing WAL mode for concurrent read/write scalability.
    """

    def __init__(self, db_path: str = "face_tracker.db"):
        self.db_path = db_path
        self._init_schema()
        self._recover_orphaned_entries() # Trigger crash recovery on startup
        logger.info(f"Database initialised at: {db_path}")

    def _conn(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL mode ensures the Flask UI can read data while the Pipeline writes data
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        """Initializes tables and performs schema migrations if necessary."""
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
            
            # Migration: Add dwell_time_seconds column to existing databases
            try:
                conn.execute("ALTER TABLE events ADD COLUMN dwell_time_seconds INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass # Column already exists

    def _recover_orphaned_entries(self):
        """
        CRASH RECOVERY: Detects if the system previously crashed without logging EXITS
        for currently tracked faces, and automatically closes out their sessions.
        """
        with self._conn() as conn:
            orphans = conn.execute('''
                SELECT face_id FROM events
                GROUP BY face_id
                HAVING SUM(CASE WHEN event_type='entry' THEN 1 ELSE 0 END) >
                       SUM(CASE WHEN event_type='exit' THEN 1 ELSE 0 END)
            ''').fetchall()
            
            if not orphans:
                return
                
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for row in orphans:
                face_id = row["face_id"]
                conn.execute(
                    "INSERT INTO events (face_id, event_type, timestamp, image_path, dwell_time_seconds) "
                    "VALUES (?, 'exit', ?, 'SYSTEM_RECOVERY', 0)",
                    (face_id, ts)
                )
        logger.info(f"Crash Recovery: Automatically closed {len(orphans)} orphaned entries.")

    def register_face(self, face_id: str, embedding: np.ndarray, crop_path: str = None) -> bool:
        """Registers a newly discovered unique face and increments the global visitor counter."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emb_json = json.dumps(embedding.tolist())
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO faces (face_id, embedding, first_seen, crop_path) VALUES (?, ?, ?, ?)",
                    (face_id, emb_json, ts, crop_path)
                )
                conn.execute(
                    "UPDATE visitor_stats SET unique_visitors=unique_visitors+1, last_updated=? WHERE id=1",
                    (ts,)
                )
            logger.info(f"DB: Registered {face_id}")
            return True
        except sqlite3.IntegrityError:
            return False

    def update_embedding(self, face_id: str, embedding: np.ndarray):
        """Updates a known face's mathematical representation to adapt to pose changes."""
        emb_json = json.dumps(embedding.tolist())
        with self._conn() as conn:
            conn.execute("UPDATE faces SET embedding=? WHERE face_id=?", (emb_json, face_id))

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Restores the active gallery from disk."""
        with self._conn() as conn:
            rows = conn.execute("SELECT face_id, embedding FROM faces").fetchall()
        return [(row["face_id"], np.array(json.loads(row["embedding"]), dtype=np.float32)) for row in rows]

    def log_event(self, face_id: str, event_type: str, image_path: str = None) -> tuple[int, int]:
        """
        Logs an entry or exit event. 
        If the event is an 'exit', it dynamically calculates the session Dwell Time.
        Returns: (row_id, dwell_time_seconds)
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dwell_time = 0

        with self._conn() as conn:
            # DWELL TIME LOGIC: Find the exact timestamp of this person's most recent entry
            if event_type == 'exit':
                last_entry = conn.execute(
                    "SELECT timestamp FROM events WHERE face_id=? AND event_type='entry' ORDER BY timestamp DESC LIMIT 1",
                    (face_id,)
                ).fetchone()
                
                if last_entry:
                    fmt = "%Y-%m-%d %H:%M:%S"
                    t1 = datetime.strptime(last_entry["timestamp"], fmt)
                    t2 = datetime.strptime(ts, fmt)
                    dwell_time = int((t2 - t1).total_seconds())

            cur = conn.execute(
                "INSERT INTO events (face_id, event_type, timestamp, image_path, dwell_time_seconds) "
                "VALUES (?, ?, ?, ?, ?)",
                (face_id, event_type, ts, image_path, dwell_time)
            )
        return cur.lastrowid, dwell_time

    def get_unique_visitor_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT unique_visitors FROM visitor_stats WHERE id=1").fetchone()
        return row["unique_visitors"] if row else 0

    def get_events_summary(self, limit: int = 200) -> list[dict]:
        """Fetches joined event data for the UI dashboard."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT e.id, e.face_id, e.event_type, e.timestamp, e.image_path, e.dwell_time_seconds,
                       f.first_seen, f.crop_path AS registered_crop
                FROM events e LEFT JOIN faces f ON e.face_id = f.face_id
                ORDER BY e.timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_all_faces(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT face_id, first_seen, crop_path FROM faces ORDER BY first_seen DESC").fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Aggregates high-level statistics for the UI header metrics."""
        with self._conn() as conn:
            v = conn.execute("SELECT unique_visitors FROM visitor_stats WHERE id=1").fetchone()["unique_visitors"]
            e = conn.execute("SELECT COUNT(*) AS c FROM events WHERE event_type='entry'").fetchone()["c"]
            x = conn.execute("SELECT COUNT(*) AS c FROM events WHERE event_type='exit'").fetchone()["c"]
            f = conn.execute("SELECT COUNT(*) AS c FROM faces").fetchone()["c"]
            d = conn.execute("SELECT AVG(dwell_time_seconds) AS a FROM events WHERE event_type='exit' AND dwell_time_seconds > 0").fetchone()
            avg_dwell = int(d["a"]) if d and d["a"] else 0
            
        return {
            "unique_visitors": v, "total_entries": e, "total_exits": x,
            "registered_faces": f, "avg_dwell_seconds": avg_dwell
        }

