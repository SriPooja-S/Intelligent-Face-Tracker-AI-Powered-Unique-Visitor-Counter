"""
pipeline.py
-----------
Master orchestration pipeline for the Face Tracker.
Combines detection, tracking, embedding, identity resolution, and hardware profiling.
"""

import logging
from collections import defaultdict, Counter
import cv2
import numpy as np
import threading
import csv
import time
import os
from datetime import datetime

from config_loader import Config
from detector import FaceDetector
from embedder import FaceEmbedder
from face_registry import FaceRegistry
from track_state_manager import TrackStateManager
from event_logger import EventLogger
from database import Database

logger = logging.getLogger(__name__)

# Discard tiny bounding boxes that are too small for accurate facial embeddings
MIN_DETECTION_AREA = 1500  
# Identity stabilization: collect N frames of data before confirming identity
VOTE_WINDOW = 3

def _build_tracker(max_age, min_hits, iou_threshold):
    """Attempts to initialize DeepSORT, falls back to SimpleTracker on failure."""
    try:
        from tracker import FaceTracker
        t = FaceTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        t.update([], np.zeros((4, 4, 3), dtype=np.uint8))
        return t
    except Exception as ex:
        from simple_tracker import SimpleTracker
        return SimpleTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)


class FaceTrackerPipeline:
    """End-to-end video processing pipeline."""

    def __init__(self, config: Config):
        self.cfg = config
        self._frame_counter = 0
        self._skip_n = config.detection.get("frame_skip", 3)
        
        # State management dicts mapping Tracker IDs to Face Embed IDs
        self._tid_to_face: dict[int, str] = {}
        self._track_votes: dict[int, list[str]] = defaultdict(list)
        self._registered_tids: set[int] = set()

        # Initialize core AI components
        self.detector = FaceDetector(
            model_path=config.detection.get("yolo_model", "yolov8n-face.pt"),
            confidence=config.detection.get("confidence_threshold", 0.45),
            input_size=config.detection.get("input_size", 640),
        )
        
        strict_thresh = config.recognition.get("similarity_threshold", 0.30)
        soft_thresh = config.recognition.get("soft_threshold", 0.25)

        self.embedder = FaceEmbedder(
            model_name=config.recognition.get("model_name", "buffalo_l"),
            det_size=tuple(config.recognition.get("det_size", [640, 640])),
            similarity_threshold=strict_thresh,
        )

        self.registry = FaceRegistry(strict_threshold=strict_thresh, soft_threshold=soft_thresh)

        # Initialize logic controllers
        self.tracker = _build_tracker(
            max_age=config.tracking.get("max_age", 30),
            min_hits=config.tracking.get("min_hits", 2),
            iou_threshold=config.tracking.get("iou_threshold", 0.3),
        )
        self.db = Database(db_path=config.database.get("path", "face_tracker.db"))
        self.event_logger = EventLogger(
            log_dir=config.logging.get("log_dir", "logs"),
            crop_size=tuple(config.logging.get("crop_size", [112, 112])),
            save_crops=config.logging.get("save_face_crops", True),
        )
        self.state_manager = TrackStateManager(
            exit_patience_frames=config.tracking.get("exit_patience_frames", 15)
        )

        # Pre-load known identities
        existing = self.db.get_all_embeddings()
        self.registry.load_from_db(existing)
        
        # --- BACKGROUND: HARDWARE COMPUTE PROFILER ---
        self.cpu_usage = 0.0
        self.ram_usage = 0.0
        self.profiler_running = True
        
        log_dir = config.logging.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.prof_filepath = os.path.join(log_dir, "compute_profile.csv")
        self.prof_file = open(self.prof_filepath, "w", newline='')
        self.prof_writer = csv.writer(self.prof_file)
        self.prof_writer.writerow(["Timestamp", "CPU_Percent", "RAM_Percent"])
        
        # Launch non-blocking profiler thread
        self.prof_thread = threading.Thread(target=self._profiler_loop, daemon=True)
        self.prof_thread.start()

    def _profiler_loop(self):
        """Monitors and records CPU/RAM usage safely in the background."""
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False
            logger.warning("psutil missing. Compute profiler disabled. Run: pip install psutil")

        while self.profiler_running:
            if has_psutil:
                self.cpu_usage = psutil.cpu_percent(interval=1)
                self.ram_usage = psutil.virtual_memory().percent
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.prof_writer.writerow([ts, self.cpu_usage, self.ram_usage])
                self.prof_file.flush()
            else:
                time.sleep(2)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Core per-frame logic: Detect -> Track -> Identify -> Log Events."""
        self._frame_counter += 1
        run_detection = (self._frame_counter == 1 or self._frame_counter % (self._skip_n + 1) == 0)

        raw_detections = []
        if run_detection:
            raw_detections = self.detector.detect(frame)

        # Strip invalid detections based on physical size
        detections = [d for d in raw_detections if (d[2] - d[0]) * (d[3] - d[1]) >= MIN_DETECTION_AREA]
        tracks = self.tracker.update(detections, frame)

        active_track_ids: list[int] = []
        track_bboxes: dict[int, tuple] = {}

        for track in tracks:
            if not track.is_confirmed(): continue
            tid = int(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1 = max(0, int(ltrb[0])), max(0, int(ltrb[1]))
            x2, y2 = min(frame.shape[1], int(ltrb[2])), min(frame.shape[0], int(ltrb[3]))

            # Guard against invalid tracker bounding boxes
            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < MIN_DETECTION_AREA: continue

            track_bboxes[tid] = (x1, y1, x2, y2)
            active_track_ids.append(tid)

            # Trigger heavy embedding extraction only on detection frames
            if run_detection:
                self._process_track_identity(frame, tid, x1, y1, x2, y2)

        # Cleanup memory for tracks that have ended
        all_live_tids = {int(t.track_id) for t in tracks}
        for tid in list(self._tid_to_face):
            if tid not in all_live_tids: del self._tid_to_face[tid]
        for tid in list(self._track_votes):
            if tid not in all_live_tids: del self._track_votes[tid]

        # Process logical entries and exits
        entries, exits = self.state_manager.update(active_track_ids, self._tid_to_face, track_bboxes)

        for face_id, bbox in entries:
            crop = self._safe_crop(frame, *bbox) if bbox else None
            img_path = self.event_logger.save_face_crop(crop, face_id, "entry")
            _, _ = self.db.log_event(face_id, "entry", img_path)
            self.event_logger.log_entry(face_id, img_path)

        for face_id, last_bbox in exits:
            crop = self._safe_crop(frame, *last_bbox) if last_bbox else None
            img_path = self.event_logger.save_face_crop(crop, face_id, "exit")
            _, dwell_time = self.db.log_event(face_id, "exit", img_path)
            self.event_logger.log_exit(face_id, img_path, dwell_time) # Pass dwell_time back to logger

        if self.cfg.display.get("draw_bboxes", True):
            frame = self._draw_overlays(frame, active_track_ids, track_bboxes)

        return frame

    def flush_remaining_tracks(self):
        """Ensures system shuts down cleanly by exiting all active tracks and closing files."""
        logger.info("Flushing active tracks at end of video...")
        for face_id in self.state_manager.get_active_face_ids():
            _, dwell_time = self.db.log_event(face_id, "exit", None)
            self.event_logger.log_exit(face_id, None, dwell_time)
            
        self.profiler_running = False
        if hasattr(self, 'prof_file') and not self.prof_file.closed:
            self.prof_file.close()

    def get_unique_visitor_count(self) -> int:
        return self.db.get_unique_visitor_count()

    def _process_track_identity(self, frame, tid, x1, y1, x2, y2):
        """Matches a detected face to the database or registers a new identity."""
        crop = self._safe_crop(frame, x1, y1, x2, y2)
        if crop is None: return

        embedding = self.embedder.get_embedding(crop)
        if embedding is None: return
        self.event_logger.log_embedding_generated("UNKNOWN", 1.0) # Requested log event

        face_id, sim = self.registry.identify(embedding)

        if face_id is not None:
            # Re-Identification: Update running mean embedding to improve future recognition
            self.registry.update_embedding(face_id, embedding)
            votes = self._track_votes[tid]
            votes.append(face_id)
            if len(votes) > VOTE_WINDOW: votes.pop(0)

            winner, vote_count = Counter(votes).most_common(1)[0]
            if vote_count >= max(1, VOTE_WINDOW // 2):
                self._tid_to_face[tid] = winner
        else:
            # Registration Gates: Prevent duplicates caused by new angles
            if self.registry.is_likely_known(embedding): return
            if tid in self._registered_tids: return

            face_id = self.registry.register_new(embedding)
            self._registered_tids.add(tid)
            crop_path = self.event_logger.save_face_crop(crop, face_id, "registered")
            self.db.register_face(face_id, embedding, crop_path)
            self.event_logger.log_registration(face_id, crop_path)
            
            self._tid_to_face[tid] = face_id
            self._track_votes[tid] = [face_id] * VOTE_WINDOW

    def _safe_crop(self, frame, x1, y1, x2, y2):
        """Returns bounded image array preventing out-of-bounds indexing."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10: return None
        return frame[y1:y2, x1:x2].copy()

    def _draw_overlays(self, frame, active_tids, track_bboxes):
        """Renders bounding boxes and UI Heads-Up Display onto the video frame."""
        unique_count = self.get_unique_visitor_count()
        for tid in active_tids:
            bbox = track_bboxes.get(tid)
            if not bbox: continue
            x1, y1, x2, y2 = bbox
            face_id = self._tid_to_face.get(tid)
            color = (0, 220, 120) if face_id else (200, 140, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = face_id[-14:] if face_id else f"track_{tid}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            
        # UI HUD Base
        cv2.rectangle(frame, (0, 0), (310, 88), (0, 0, 0), -1) 
        cv2.putText(frame, f"Unique visitors: {unique_count}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2)
        cv2.putText(frame, f"Frame: {self._frame_counter}  Tracks: {len(active_tids)}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        # Profiler Metrics
        cv2.putText(frame, f"CPU: {self.cpu_usage}% | RAM: {self.ram_usage}%", (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
        return frame