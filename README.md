# Intelligent Face Tracker — AI-Powered Unique Visitor Counter

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-green)](https://ultralytics.com)
[![InsightFace](https://img.shields.io/badge/Recognition-InsightFace%20ArcFace-orange)](https://insightface.ai)
[![Flask](https://img.shields.io/badge/Dashboard-Flask-lightgrey)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Project Overview

**Problem:** Count unique visitors from a video stream without double-counting re-entries or the same face appearing across multiple frames.

**Solution:** A modular, production-grade AI pipeline that detects faces using YOLOv8, generates 512-d embeddings via InsightFace ArcFace, tracks identities with DeepSORT, and fires exactly one `ENTRY` and one `EXIT` event per visit session — all logged to file, filesystem, and a SQLite database.

**Expected System Outputs:**

| Output | Description |
|---|---|
| `logs/events.log` | Strictly formatted log of every `ENTRY`, `EXIT`, `REGISTERED`, and `EMBEDDING` event with timestamps and Face IDs |
| `logs/entries/YYYY-MM-DD/` | Cropped face images at entry time |
| `logs/exits/YYYY-MM-DD/` | Cropped face images at exit time |
| `logs/registered/YYYY-MM-DD/` | Cropped face images at first registration |
| `face_tracker.db` | SQLite database with face metadata, embeddings, and full event history |
| `logs/compute_profile.csv` | Per-second CPU/RAM samples from the background hardware profiler |
| Live Dashboard (`http://localhost:5000`) | Real-time web UI to monitor traffic, view face crops, and control the pipeline |

---

## 🛠️ Tech Stack

| Module | Technology |
|---|---|
| Face Detection | YOLOv8n-face (Ultralytics) |
| Face Recognition | InsightFace `buffalo_l` — ArcFace 512-d embeddings |
| Multi-Object Tracking | DeepSORT + IoU fallback |
| Backend & Processing | Python 3.10+ |
| Database | SQLite (WAL mode for concurrent access) |
| Web Dashboard | Flask |
| Hardware Profiling | `psutil` (daemon thread) |
| Configuration | `config.json` |
| Camera Input | Video file (`.mp4`) or RTSP stream |
| Deployment | Local / Google Colab (with `ngrok`) |

---

## ✨ Unique Features (Beyond the Problem Statement)

These enterprise-grade additions were built to make the system production-ready, not just demo-ready:

- **⏳ Dwell Time Analytics** — Calculates exactly how long each person stayed in frame per session and surfaces the "Avg Dwell Time" live on the dashboard — not just entry/exit timestamps.
- **🛡️ Graceful Crash Recovery** — If the process terminates unexpectedly (SIGKILL, power loss), orphaned open sessions are detected on the next startup and automatically closed with synthetic `EXIT` events, keeping the database fully consistent.
- **📈 Automated Compute Profiler** — A background daemon thread uses `psutil` to sample CPU and RAM every second, displays metrics on the video HUD, and writes them to `logs/compute_profile.csv` for post-run analysis.
- **🌐 Live Web Dashboard** — A built-in Flask application provides a real-time UI to start/stop the pipeline, switch between video file and RTSP stream, view live face crops, and inspect database metrics — no external tooling required.

---

## 🔒 Duplicate Prevention Strategy

Preventing double-counting is the core correctness challenge. The system uses a layered approach:

1. **Cosine Similarity Matching** — Each new embedding is compared against all registered faces. A cosine distance below `similarity_threshold` (default: `0.35`) identifies the face as a known visitor — no new UUID is created.
2. **Soft Gate** — A secondary `soft_threshold` (default: `0.28`) catches borderline near-misses: faces too similar to be new but not confidently matched are blocked from registration rather than creating a duplicate entry.
3. **Temporal Vote Buffer (3-frame)** — A face must be consistently identified over 3 consecutive frames before an identity decision is committed. This prevents flicker from head turns, partial occlusions, or lighting changes from triggering spurious registrations.
4. **Running-Mean Embedding Update** — Once registered, a face's stored embedding updates as a running mean across subsequent detections, adapting to appearance changes without replacing the identity.
5. **`TrackStateManager` Session Guard** — Even if recognition fires multiple times for the same person in one visit, the state manager enforces exactly **one** `ENTRY` and **one** `EXIT` per session at the application layer.

---

## 🧠 System Architecture & AI Planning

### Planning Summary

Built on a **modular, separation-of-concerns** architecture — each responsibility (detection, embedding, tracking, state management, logging, serving) lives in its own module. Heavy inference (YOLO + InsightFace) runs only every N frames (`frame_skip`), while DeepSORT handles bounding-box continuity in between, keeping the system real-time capable even on CPU.

### Pipeline Description

**1. Detection** — `detector.py`  
YOLOv8n-face runs every `frame_skip + 1` frames. A minimum detection area filter discards distant or low-quality faces before any embedding work.

**2. Recognition** — `embedder.py`  
InsightFace `buffalo_l` generates a 512-d ArcFace embedding per crop. The `face_recognition` library was intentionally excluded — insufficient accuracy for production re-identification.

**3. Tracking** — `tracker.py`  
DeepSORT maintains identity continuity between YOLO frames. An IoU fallback handles cases where DeepSORT drops a track.

**4. Identity Resolution** — `face_registry.py`  
Embeddings are matched via cosine similarity against all registered faces. Unmatched faces are auto-registered with a UUID. A 3-frame vote buffer and running-mean update protect against false registrations (see Duplicate Prevention).

**5. State Management** — `TrackStateManager`  
Enforces exactly **one** `ENTRY` and **one** `EXIT` per visit session regardless of frame count.

**6. Visitor Counting**  
Unique visitor count = number of distinct UUIDs in the `faces` table. Re-identification never creates a new UUID.

### Architecture Diagram

```
┌──────────────────────────────────────────────┐
│               INPUT SOURCE                   │
│    Video File (.mp4)  OR  RTSP Stream        │
└───────────────────┬──────────────────────────┘
                    │ frames
                    ▼
┌──────────────────────────────────────────────┐
│           AI VISION PIPELINE                 │
│  1. YOLOv8 Face Detection  (every N frames)  │
│  2. DeepSORT Tracking      (every frame)     │
│  3. InsightFace ArcFace Embeddings (512-d)   │
└───────────────────┬──────────────────────────┘
                    │ Face IDs + Bounding Boxes
                    ▼
┌──────────────────────────────────────────────┐
│         LOGIC & STATE MANAGEMENT             │
│  - Face Registry  (identity + dedup)         │
│  - Track State Manager (1 entry, 1 exit)     │
│  - Compute Profiler  (background daemon)     │
└────────────┬─────────────┬───────────────────┘
             │             │
             ▼             ▼
┌──────────────────┐  ┌───────────────────────┐
│  SQLite Database │  │  Filesystem Logging   │
│  - faces table   │  │  - logs/entries/      │
│  - events table  │  │  - logs/exits/        │
│  - dwell times   │  │  - logs/registered/   │
└──────────┬───────┘  │  - logs/events.log    │
           │          │  - compute_profile.csv│
           │          └───────────────────────┘
           ▼
┌──────────────────────────────────────────────┐
│          FLASK WEB DASHBOARD                 │
│  Live metrics · Event stream · Controls      │
└──────────────────────────────────────────────┘
```

### Compute Load Estimates

Frame skipping (`frame_skip: 3`) means heavy models run every 4th frame; DeepSORT handles the rest.

| Component | CPU Load | GPU Load | Approx. Inference Time |
|---|---|---|---|
| YOLOv8n detection | Medium | Medium (CUDA) | ~80ms CPU / 15ms GPU |
| InsightFace buffalo_l | Medium | Medium (CUDA) | ~50ms CPU / 10ms GPU |
| DeepSORT tracking | Low | None | ~2ms |
| Background Profiler | Negligible | None | <1ms |
| **Total (CPU only)** | **~75%** | — | **~130ms/frame** |
| **Total (T4 GPU)** | **~20%** | **~40%** | **~25ms/frame** |

*Metrics are actively verified by the built-in `compute_profile.csv` logger.*

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA for real-time performance

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/face-tracker.git
cd face-tracker

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install psutil              # Required for hardware profiling
```

> **CPU-only users:** Replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt`.

### 2. Download YOLO Weights

```bash
python3 -c "
from ultralytics import YOLO
YOLO('yolov8n-face.pt')
print('Model ready.')
"
```

InsightFace's `buffalo_l` weights (~200MB) are downloaded automatically on first run — internet access required.

### 3. Configure

Edit `config.json` to point at your video file or RTSP stream (see [Sample config.json](#-sample-configjson) below).

### 4. Run

```bash
python main.py
```

Open `http://localhost:5000` to access the live dashboard.

### Google Colab

A Colab notebook (`Face_Tracker_Colab.ipynb`) is included for cloud-based testing. Mount Google Drive, upload your video file, and run all cells. The dashboard is accessible via the `ngrok` tunnel URL printed in the output.

---

## 📄 Sample `config.json`

```json
{
  "video_source": "video_sample1.mp4",
  "rtsp_url": "rtsp://username:password@camera_ip:554/stream",
  "use_rtsp": false,
  "detection": {
    "yolo_model": "yolov8n-face.pt",
    "confidence_threshold": 0.65,
    "frame_skip": 3,
    "input_size": 640
  },
  "recognition": {
    "model_name": "buffalo_l",
    "similarity_threshold": 0.35,
    "soft_threshold": 0.28,
    "embedding_size": 512,
    "det_size": [640, 640]
  },
  "tracking": {
    "max_age": 30,
    "min_hits": 2,
    "iou_threshold": 0.3,
    "exit_patience_frames": 15
  },
  "logging": {
    "log_dir": "logs",
    "log_file": "logs/events.log",
    "log_level": "INFO",
    "save_face_crops": true,
    "crop_size": [112, 112]
  },
  "database": {
    "path": "face_tracker.db"
  },
  "display": {
    "show_window": false,
    "draw_bboxes": true
  }
}
```

**Key Parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `frame_skip` | `3` | Run YOLO every 4th frame; DeepSORT tracks all frames |
| `confidence_threshold` | `0.65` | Minimum YOLO detection confidence |
| `similarity_threshold` | `0.35` | Cosine similarity for face match (lower = stricter match) |
| `soft_threshold` | `0.28` | Below this → register as new face; above → block as likely duplicate |
| `exit_patience_frames` | `15` | Frames a track must be absent before firing an exit event |
| `min_hits` | `2` | Track confirmations required before recognising a face |
| `use_rtsp` | `false` | Set `true` to switch from video file to live RTSP stream |

---

## 📤 Output Description

### `logs/events.log` (real sample output)

```
2026-03-23 17:36:37 [INFO] [EventLogger] EVENT: REGISTERED | Face ID: face_20260323_173637_fcc90a | Timestamp: 2026-03-23 17:36:37 | Image: logs/registered/2026-03-23/face_20260323_173637_fcc90a_173637_739.jpg
2026-03-23 17:36:37 [INFO] [EventLogger] EVENT: EMBEDDING | Face ID: UNKNOWN | Best Sim: 1.0000
2026-03-23 17:36:37 [INFO] [EventLogger] EVENT: EMBEDDING | Face ID: UNKNOWN | Best Sim: 1.0000
2026-03-23 17:36:37 [INFO] [track_state_manager] ENTRY queued: face_id=face_20260323_173630_9ea6b0
2026-03-23 17:36:37 [INFO] [track_state_manager] ENTRY queued: face_id=face_20260323_173637_fcc90a
2026-03-23 17:36:37 [INFO] [EventLogger] EVENT: ENTRY | Face ID: face_20260323_173630_9ea6b0 | Timestamp: 2026-03-23 17:36:37 | Image: logs/entries/2026-03-23/face_20260323_173630_9ea6b0_173637_891.jpg
2026-03-23 17:36:37 [INFO] [EventLogger] EVENT: ENTRY | Face ID: face_20260323_173637_fcc90a | Timestamp: 2026-03-23 17:36:37 | Image: logs/entries/2026-03-23/face_20260323_173637_fcc90a_173637_910.jpg
```

### Database Tables
- **`faces`** — UUID, first-seen timestamp, embedding vector, total visit count.
- **`events`** — Event type, Face ID, timestamp, crop image path, session dwell time.

### Dashboard Panels
- Total unique visitors (live counter)
- Active tracks in current frame
- Recent entry/exit face crops
- Average dwell time per session
- CPU / RAM utilisation (from background profiler)

---

## 🎬 Demo Video

```
**[Click here to watch the full demo on ScreenApp](https://screenapp.io/app/v/1N2yn56c8z)**
```

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard](screenshots/1.png)

### Face Detection & Tracking
![Detection](screenshots/2.png)

### Face Recognition (ArcFace Embeddings)
![Recognition](screenshots/3.png)

### Entry Logs
![Entries](screenshots/4.png)

### Registered Faces
![Registered Faces](screenshots/5.png)

### Unique Visitor Stats
![Stats](screenshots/6.png)

---

## 🔍 Code Highlights

| Module | Role | Key Decision |
|---|---|---|
| `detector.py` | YOLOv8n-face inference | Minimum area filter cuts distant/blurry faces before embedding |
| `embedder.py` | InsightFace buffalo_l | 512-d ArcFace chosen over `face_recognition` for production accuracy |
| `tracker.py` | DeepSORT + IoU fallback | IoU fallback prevents identity loss when DeepSORT drops a track |
| `face_registry.py` | Identity matching & registration | Running-mean embedding update + vote buffer prevent false registrations |
| `event_logger.py` | Structured logging | Third-party logs (YOLO/Flask) explicitly silenced; only system events written |
| `database.py` | SQLite in WAL mode | WAL mode allows concurrent dashboard reads during write-heavy processing |
| `TrackStateManager` | Session state machine | Enforces exactly 1 entry + 1 exit per session regardless of frame count |

**Key Optimisations:**
- `frame_skip: 3` reduces heavy-model calls by 75% while DeepSORT maintains smooth tracking.
- Crash recovery via orphaned-session detection on startup keeps the DB consistent after force-quits.
- Compute profiler runs as a daemon thread with negligible overhead (<1ms per sample).

---

## 📝 Assumptions Made

1. **Camera angle:** System is tuned for high-mounted, top-down, or slightly angled cameras. Detection thresholds (`MIN_DETECTION_AREA = 1500`, confidence `0.65`) reflect this.
2. **Re-entry dwell time:** If the same person leaves and re-enters, the unique count is not incremented, but a fresh dwell time is calculated for the new session.
3. **Crash recovery scope:** Forceful termination (SIGKILL, power loss) marks all open sessions as exited on the next startup. Graceful shutdown always closes sessions cleanly.
4. **First-run internet access:** InsightFace `buffalo_l` weights (~200MB) are downloaded automatically on the first run.
5. **Single-camera setup:** The system is designed for one input source at a time (video file or single RTSP stream).

---

> This project is a part of a hackathon run by https://katomaran.com
