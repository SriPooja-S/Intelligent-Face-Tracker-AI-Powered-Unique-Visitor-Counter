"""
verify.py
---------
Run this after main.py to confirm data was stored correctly.

Usage:
    python verify.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from database import Database
import sqlite3

DB_PATH = "face_tracker.db"
LOGS_DIR = "logs"

print("\n" + "="*60)
print("FACE TRACKER — POST-RUN VERIFICATION")
print("="*60)

# 1. Check DB exists
if not os.path.exists(DB_PATH):
    print(f"\n✗ Database not found: {DB_PATH}")
    print("  Run main.py first to process a video.")
    sys.exit(1)

db = Database(DB_PATH)
stats = db.get_stats()
faces = db.get_all_faces()
events = db.get_events_summary(limit=200)

print(f"\n{'DATABASE':}")
print(f"  File              : {DB_PATH} ({os.path.getsize(DB_PATH)/1024:.1f} KB)")
print(f"  Unique visitors   : {stats['unique_visitors']}")
print(f"  Total entries     : {stats['total_entries']}")
print(f"  Total exits       : {stats['total_exits']}")
print(f"  Registered faces  : {len(faces)}")

# 2. Check images exist
print(f"\n{'LOG IMAGES':}")
total_images = 0
missing_images = 0
for e in events:
    if e["image_path"]:
        total_images += 1
        if not os.path.exists(e["image_path"]):
            missing_images += 1

print(f"  Event image paths in DB   : {total_images}")
print(f"  Missing on disk           : {missing_images}")
print(f"  Images found on disk      : {total_images - missing_images}")

# 3. Count actual files in logs/
print(f"\n{'LOG FOLDERS':}")
for subfolder in ["entries", "exits", "registered"]:
    folder = os.path.join(LOGS_DIR, subfolder)
    if os.path.exists(folder):
        count = sum(
            len(files) for _, _, files in os.walk(folder)
            if files
        )
        print(f"  logs/{subfolder}/   : {count} image(s)")
    else:
        print(f"  logs/{subfolder}/   : folder missing")

# 4. Show events.log line count
log_file = os.path.join(LOGS_DIR, "events.log")
if os.path.exists(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    print(f"\n{'EVENTS LOG':}")
    print(f"  events.log lines  : {len(lines)}")
    print(f"  Last 5 lines:")
    for line in lines[-5:]:
        print(f"    {line.rstrip()}")

# 5. Show registered faces
if faces:
    print(f"\n{'REGISTERED FACES':}")
    for f in faces[:10]:
        crop_ok = "✓" if f["crop_path"] and os.path.exists(f["crop_path"]) else "✗"
        print(f"  [{crop_ok} crop] {f['face_id'][:40]} | first seen: {f['first_seen']}")

# 6. Entry/exit balance check
entries_set = {e["face_id"] for e in events if e["event_type"] == "entry"}
exits_set   = {e["face_id"] for e in events if e["event_type"] == "exit"}
unmatched   = entries_set - exits_set

print(f"\n{'ENTRY/EXIT BALANCE':}")
print(f"  Faces with entry  : {len(entries_set)}")
print(f"  Faces with exit   : {len(exits_set)}")
if unmatched:
    print(f"  Without exit yet  : {len(unmatched)} (normal if video just ended)")
else:
    print(f"  All entries have matching exits ✓")

print("\n" + "="*60)
print("Verification complete.")
print("="*60 + "\n")
