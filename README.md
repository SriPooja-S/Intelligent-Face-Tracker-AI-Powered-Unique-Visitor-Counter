# Intelligent Face Tracker — AI-Powered Unique Visitor Counter

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-green)](https://ultralytics.com)
[![InsightFace](https://img.shields.io/badge/Recognition-InsightFace%20ArcFace-orange)](https://insightface.ai)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **This project is a part of a hackathon run by https://katomaran.com**

---

## 🎬 Demo Video

> 📹 **[Click here to watch the full demo on YouTube/Loom](https://www.loom.com/share/f3dc80ce9b234a46b7ccaf532592a393)**
>
> *

---

## 📋 Overview

An AI-driven real-time face tracking system that:

- Detects faces in a video file or live RTSP camera stream using **YOLOv8**
- Generates 512-dimensional **ArcFace embeddings** via InsightFace for each detected face
- Tracks faces across frames using **DeepSORT** (with IoU-based SimpleTracker fallback)
- **Auto-registers** each new unique face with a UUID on first detection
- Logs every **entry and exit** event with a timestamped cropped image
- Maintains an accurate **unique visitor count** — re-identification never increments the count
- Stores all metadata in **SQLite** and images in a structured folder hierarchy
- Provides a **live web dashboard** (Flask + Cloudflare tunnel for Colab) with Start / Pause / Resume / Stop / Clear controls

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT SOURCE                            │
│         Video File (.mp4)  OR  RTSP Camera Stream          │
└───────────────────────┬─────────────────────────────────────┘
                        │ frames
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  DETECTION LAYER  (every N frames — configurable)           │
│  YOLOv8n → bounding boxes for persons/faces                 │
│  Min area filter → discard tiny/distant detections          │
└───────────────────────┬─────────────────────────────────────┘
                        │ detections
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  TRACKING LAYER  (every frame)                              │
│  DeepSORT / SimpleTracker → stable track IDs via Kalman     │
│  Tracks persist across skip frames using motion prediction  │
└───────────────────────┬─────────────────────────────────────┘
                        │ confirmed tracks
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  RECOGNITION LAYER  (on detection frames)                   │
│  InsightFace buffalo_l → 512-d ArcFace embedding            │
│  Multi-embedding gallery (up to 5 poses per face)           │
│  Running mean update on re-identification                    │
│  Temporal vote buffer (3 frames) for stable identity        │
│  Strict threshold 0.35 + soft gate 0.28 for dedup           │
└───────────────────────┬─────────────────────────────────────┘
                        │ face identity
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  REGISTRATION  (new faces only)                             │
│  Assign unique face_id → store embedding → save crop image  │
│  Per-track lock prevents double registration                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  ENTRY / EXIT STATE MACHINE                                 │
│  TrackStateManager → fires exactly ONE entry + ONE exit     │
│  Exit patience: 15 frames before confirming exit            │
│  End-of-video flush: closes all open tracks                 │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │
             ▼                          ▼
┌─────────────────────┐    ┌────────────────────────────────┐
│  SQLite Database    │    │  Filesystem Logs               │
│  faces table        │    │  logs/entries/YYYY-MM-DD/      │
│  events table       │    │  logs/exits/YYYY-MM-DD/        │
│  visitor_stats      │    │  logs/registered/YYYY-MM-DD/   │
│  WAL journal mode   │    │  logs/events.log               │
└─────────────────────┘    └────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  FLASK DASHBOARD  (runs alongside pipeline)                 │
│  Live stats: unique visitors, entries, exits                │
│  Tabs: Entries | Exits | Registered Faces | All Events      │
│  Controls: Start | Pause | Resume | Stop | Clear Data       │
│  Cloudflare tunnel for Google Colab access                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
face_tracker/
│
├── main.py                      # Single entry point — Flask + pipeline
├── verify.py                    # Post-run verification report
├── inspect_db.py                # CLI database inspector
├── colab_runner.ipynb           # Google Colab notebook
├── requirements.txt
│
├── config/
│   └── config.json              # All configurable parameters
│
├── src/
│   ├── config_loader.py         # Config file parser
│   ├── detector.py              # YOLOv8 face/person detector
│   ├── embedder.py              # InsightFace ArcFace embedder
│   ├── tracker.py               # DeepSORT tracker wrapper
│   ├── simple_tracker.py        # Pure Python IoU fallback tracker
│   ├── face_registry.py         # Multi-embedding in-memory gallery
│   ├── track_state_manager.py   # Entry/exit state machine
│   ├── event_logger.py          # File system logging + events.log
│   ├── database.py              # SQLite persistence layer
│   └── pipeline.py              # Master per-frame orchestrator
│
├── logs/
│   ├── events.log               # All critical system events (mandatory)
│   ├── entries/YYYY-MM-DD/      # Face crops at entry moment
│   ├── exits/YYYY-MM-DD/        # Face crops at exit moment
│   └── registered/YYYY-MM-DD/  # Face crops at first registration
│
└── face_tracker.db              # SQLite database (auto-created)
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional but recommended) NVIDIA GPU with CUDA for faster inference

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/face-tracker.git
cd face-tracker
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **CPU-only machines:** Replace `onnxruntime-gpu` with `onnxruntime` in requirements.txt before installing.

### 4. Download the YOLO model weights (one-time)

```bash
python3 -c "
from ultralytics import YOLO
import shutil
YOLO('yolov8n.pt')
shutil.copy('yolov8n.pt', 'yolov8n-face.pt')
print('Model ready.')
"
```

### 5. Place your video file in the project root

Download the sample video from the provided Google Drive link and save it as `video_sample1.mp4` in the project root.

### 6. Run the application

```bash
# Start dashboard + auto-process the configured video file
python main.py

# Open dashboard in browser
open http://localhost:5000
```

**Or use the web UI to select source at runtime:**
```bash
python main.py        # opens dashboard, use control panel to pick source
```

### 7. Switch between video file and RTSP camera

**Via the web UI:** Select "Video File" or "RTSP Camera" from the dropdown, enter the source, click Start.

**Via command line:**
```bash
# Video file
python main.py --source video_sample1.mp4

# RTSP camera
python main.py --source rtsp://admin:password@192.168.1.100:554/stream
```

### 8. Verify results after processing

```bash
python verify.py       # full verification report
python inspect_db.py   # browse database contents
```

---

## ☁️ Google Colab Setup

Use `colab_runner.ipynb` for GPU-accelerated processing.

1. Upload the project folder to Google Drive
2. Open `colab_runner.ipynb` in Colab
3. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
4. Run cells 1–6 for setup
5. Run Cell 7 — prints a public Cloudflare URL for the dashboard
6. Open the URL, use the control panel to start processing

---

## 🔧 Configuration (`config/config.json`)

```json
{
  "video_source": "video_sample1.mp4",
  "rtsp_url": "rtsp://username:password@camera_ip:554/stream",
  "use_rtsp": false,

  "detection": {
    "yolo_model": "yolov8n.pt",
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

  "database": { "path": "face_tracker.db" },

  "display": {
    "show_window": false,
    "window_name": "Face Tracker",
    "draw_bboxes": true
  }
}
```

### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `frame_skip` | `3` | Run YOLO every 4th frame; DeepSORT tracks all frames |
| `confidence_threshold` | `0.65` | Minimum YOLO detection confidence |
| `similarity_threshold` | `0.35` | Cosine similarity for face match (lower = easier match) |
| `soft_threshold` | `0.28` | Below this → register as new face; above → block as likely duplicate |
| `exit_patience_frames` | `15` | Frames absent before firing exit event |
| `min_hits` | `2` | Track confirmations before recognising a face |

---

## 🤖 AI Planning Document

### Goal
Count the number of unique human faces appearing in a video stream or live camera feed, with each unique face logged exactly once per appearance (entry + exit pair).

### Feature List

| # | Feature | Module | Status |
|---|---|---|---|
| 1 | Face/person detection per frame | `detector.py` (YOLOv8) | ✅ |
| 2 | 512-d ArcFace embedding extraction | `embedder.py` (InsightFace) | ✅ |
| 3 | Multi-face tracking across frames | `tracker.py` (DeepSORT) | ✅ |
| 4 | Auto-registration of new faces | `face_registry.py` + `database.py` | ✅ |
| 5 | Re-identification (no double count) | Cosine similarity + multi-embedding gallery | ✅ |
| 6 | Exactly-one entry event per face | `track_state_manager.py` | ✅ |
| 7 | Exactly-one exit event per face | `track_state_manager.py` | ✅ |
| 8 | Cropped face image storage | `event_logger.py` | ✅ |
| 9 | Structured `events.log` file | `event_logger.py` | ✅ |
| 10 | SQLite persistence (WAL mode) | `database.py` | ✅ |
| 11 | Configurable frame skip | `config.json` | ✅ |
| 12 | Unique visitor count in DB | `database.py` | ✅ |
| 13 | Data persistence across video runs | Gallery restored from DB at startup | ✅ |
| 14 | RTSP live stream support | `main.py` | ✅ |
| 15 | Live web dashboard | Flask in `main.py` | ✅ (bonus) |
| 16 | Pause / Resume / Stop / Clear | Flask routes in `main.py` | ✅ (bonus) |
| 17 | Public URL via Cloudflare tunnel | `main.py --colab` | ✅ (bonus) |

### Compute Load Estimate

| Component | CPU Load | GPU Load | Approx. time/frame |
|---|---|---|---|
| YOLOv8n detection | Medium | Medium (CUDA) | 80ms CPU / 15ms GPU |
| InsightFace buffalo_l | Medium | Medium (CUDA) | 50ms CPU / 10ms GPU |
| DeepSORT tracking | Low | None | 2ms |
| SQLite I/O | Negligible | None | <1ms |
| **Total (CPU only)** | **60–80%** | — | ~130ms/frame |
| **Total (T4 GPU)** | ~15% | ~40% | ~25ms/frame |

*Frame skip of 3 means YOLO+InsightFace run every 4th frame, reducing effective load by 75%.*

### Duplicate Prevention Strategy

Three-layer defence against counting the same person twice:

1. **Strict threshold (0.35):** Only register if best gallery similarity < 0.35
2. **Soft gate (0.28):** Block registration if any face scores > 0.28 (catches near-misses)
3. **Per-track lock:** Once a DeepSORT track_id has registered, it cannot register again
4. **Temporal voting:** Collect 3 frames of identity votes before committing face_id
5. **Running mean update:** Re-identification updates stored embedding toward current observation, adapting to pose changes

---

## 📊 Sample Output

### events.log (excerpt)
```
2026-03-22 18:47:01 [INFO] [EventLogger] REGISTERED | face_id=face_20260322_184701_1875fd | image=logs/registered/2026-03-22/...jpg
2026-03-22 18:47:01 [INFO] [EventLogger] ENTRY | face_id=face_20260322_184701_1875fd | image=logs/entries/2026-03-22/...jpg
2026-03-22 18:47:14 [INFO] [EventLogger] EXIT  | face_id=face_20260322_184701_1875fd | image=logs/exits/2026-03-22/...jpg
```

### Database query
```
Unique visitors  : 25
Registered faces : 25
Entry events     : 52
Exit events      : 52
```

---

## 🛠️ Tech Stack

| Module | Technology | Justification |
|---|---|---|
| Face detection | YOLOv8n | Real-time speed, official face/person weights |
| Face recognition | InsightFace buffalo_l (ArcFace) | SOTA accuracy, GPU-ready via ONNX, avoids `face_recognition` library |
| Tracking | DeepSORT + SimpleTracker fallback | Stable cross-frame IDs, automatic API version compatibility |
| Database | SQLite (WAL mode) | Zero config, relational, resilient to interruptions |
| Configuration | JSON | As specified in problem statement |
| Logging | Python logging + filesystem | events.log + dated folder structure |
| Web UI | Flask | Lightweight, server-side rendering, no JS fetch issues |
| Colab tunnel | Cloudflare Quick Tunnel | Free, no account, no token required |

---

## 📝 Assumptions

1. The same physical person re-entering after a gap is recognised from their stored embedding and does **not** increment the unique count.
2. `similarity_threshold: 0.35` works well for InsightFace buffalo_l. Adjust between 0.28–0.45 based on camera conditions.
3. InsightFace buffalo_l (~200 MB) auto-downloads on first run — internet access required once.
4. `show_window: false` is the correct default for headless environments (Colab, servers).
5. YOLO detects full-body bounding boxes; InsightFace finds and aligns the face within each crop.
6. SQLite is sufficient for single-process use; for multi-camera production, swap to PostgreSQL.

---

## 🗂️ GitHub Repository Contents Checklist

```
✅ main.py
✅ verify.py
✅ inspect_db.py
✅ colab_runner.ipynb
✅ requirements.txt
✅ README.md
✅ config/config.json
✅ src/config_loader.py
✅ src/detector.py
✅ src/embedder.py
✅ src/tracker.py
✅ src/simple_tracker.py
✅ src/face_registry.py
✅ src/track_state_manager.py
✅ src/event_logger.py
✅ src/database.py
✅ src/pipeline.py
✅ logs/.gitkeep          (empty placeholder — actual logs excluded via .gitignore)
✅ .gitignore
```

---

## 🔗 Live Demo

> Deployed on Hugging Face Spaces: **[YOUR_HF_SPACE_LINK_HERE]**

---

> This project is a part of a hackathon run by https://katomaran.com
