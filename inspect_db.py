"""
inspect_db.py
-------------
View the contents of face_tracker.db in a readable format.

Usage:
    python inspect_db.py              # show everything
    python inspect_db.py --faces      # only registered faces
    python inspect_db.py --events     # only events table
    python inspect_db.py --stats      # only summary counts
    python inspect_db.py --sql "SELECT * FROM events LIMIT 5"  # custom query

This script does NOT require the venv or any imports beyond sqlite3,
which is built into Python. You can run it any time, even while
main.py is running (SQLite WAL mode allows concurrent reads).
"""

import sqlite3
import argparse
import os
import sys

DB_PATH = "face_tracker.db"


def connect():
    if not os.path.exists(DB_PATH):
        print(f"\n✗ Database file not found: {DB_PATH}")
        print("  Run main.py first to generate data.\n")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def print_table(rows, title):
    if not rows:
        print(f"\n  [{title}] — no rows found\n")
        return
    rows = [dict(r) for r in rows]
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c] or "")) for r in rows)) for c in cols}
    widths = {c: min(w, 50) for c, w in widths.items()}  # cap at 50 chars

    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(c.upper().ljust(widths[c]) for c in cols) + " |"

    print(f"\n  [{title}] — {len(rows)} row(s)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        row_str = "| " + " | ".join(
            str(r[c] or "")[:widths[c]].ljust(widths[c]) for c in cols
        ) + " |"
        print(row_str)
    print(sep)


def show_stats(conn):
    print("\n" + "=" * 55)
    print("  FACE TRACKER DATABASE — SUMMARY")
    print("=" * 55)

    size_kb = os.path.getsize(DB_PATH) / 1024
    print(f"  File: {DB_PATH}  ({size_kb:.1f} KB)")

    v = conn.execute("SELECT unique_visitors FROM visitor_stats WHERE id=1").fetchone()
    print(f"\n  Unique visitors  : {v['unique_visitors'] if v else 0}")

    e = conn.execute("SELECT COUNT(*) as c FROM events WHERE event_type='entry'").fetchone()
    x = conn.execute("SELECT COUNT(*) as c FROM events WHERE event_type='exit'").fetchone()
    f = conn.execute("SELECT COUNT(*) as c FROM faces").fetchone()

    print(f"  Registered faces : {f['c']}")
    print(f"  Entry events     : {e['c']}")
    print(f"  Exit events      : {x['c']}")
    print("=" * 55)


def show_faces(conn):
    rows = conn.execute(
        "SELECT face_id, first_seen, crop_path FROM faces ORDER BY first_seen DESC"
    ).fetchall()
    print_table(rows, "REGISTERED FACES")


def show_events(conn, limit=50):
    rows = conn.execute(f"""
        SELECT e.id, e.face_id, e.event_type, e.timestamp, e.image_path
        FROM events e
        ORDER BY e.timestamp DESC
        LIMIT {limit}
    """).fetchall()
    print_table(rows, f"EVENTS (latest {limit})")


def run_custom(conn, sql):
    try:
        rows = conn.execute(sql).fetchall()
        print_table(rows, f"CUSTOM QUERY")
    except Exception as ex:
        print(f"\n✗ SQL error: {ex}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect face_tracker.db")
    parser.add_argument("--faces",  action="store_true", help="Show faces table")
    parser.add_argument("--events", action="store_true", help="Show events table")
    parser.add_argument("--stats",  action="store_true", help="Show summary counts")
    parser.add_argument("--sql",    type=str, help="Run a custom SQL query")
    args = parser.parse_args()

    conn = connect()

    # Default: show everything
    if not any([args.faces, args.events, args.stats, args.sql]):
        show_stats(conn)
        show_faces(conn)
        show_events(conn)
    else:
        if args.stats:  show_stats(conn)
        if args.faces:  show_faces(conn)
        if args.events: show_events(conn)
        if args.sql:    run_custom(conn, args.sql)

    conn.close()
    print()


if __name__ == "__main__":
    main()