"""
event_logger.py
---------------
Handles all file-system logging for the Face Tracker application.
Responsible for saving cropped images and maintaining the strictly formatted events.log.
"""

import logging
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------ #
# Root Logger Configuration
# ------------------------------------------------------------------ #

def setup_logging(log_file: str = "logs/events.log", log_level: str = "INFO"):
    """
    Configures the root logger and silences third-party library spam.
    Must be called exactly once at application startup.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Katomaran required timestamp prefix format
    fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler() # Also print to console
        ],
        force=True # Overrides YOLO/Flask default loggers
    )

    # Suppress verbose outputs from external libraries to keep events.log clean
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('deep_sort_realtime').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)


# ------------------------------------------------------------------ #
# EventLogger Class
# ------------------------------------------------------------------ #

class EventLogger:
    """Saves face crop images and emits structured log entries."""

    def __init__(self, log_dir: str = "logs", crop_size: tuple = (112, 112), save_crops: bool = True):
        self.log_dir = Path(log_dir)
        self.crop_size = crop_size
        self.save_crops = save_crops
        self.logger = logging.getLogger("EventLogger")

    def _dated_dir(self, subfolder: str) -> Path:
        """Dynamically generates and returns a directory path structured by current date."""
        today = datetime.now().strftime("%Y-%m-%d")
        path = self.log_dir / subfolder / today
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_face_crop(self, face_crop: np.ndarray, face_id: str, event_type: str) -> str | None:
        """
        Resizes and saves a BGR numpy image array to the local filesystem.
        Returns the absolute filepath or None if saving is disabled/failed.
        """
        if not self.save_crops or face_crop is None or face_crop.size == 0:
            return None

        # Route to appropriate subfolder based on event type
        subdir_map = {"entry": "entries", "exit": "exits", "registered": "registered"}
        subdir = subdir_map.get(event_type, "entries")
        directory = self._dated_dir(subdir)

        # Generate unique filename using timestamp
        ts = datetime.now().strftime("%H%M%S_%f")[:10]
        filename = f"{face_id}_{ts}.jpg"
        filepath = directory / filename

        crop_resized = cv2.resize(face_crop, self.crop_size)
        cv2.imwrite(str(filepath), crop_resized)
        return str(filepath)

    # ---------------------------------------------------------------- #
    # Strict Output Formatting (Problem Statement Compliance)
    # ---------------------------------------------------------------- #

    def log_entry(self, face_id: str, image_path: str | None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"EVENT: ENTRY | Face ID: {face_id} | Timestamp: {ts} | Image: {image_path or 'none'}")

    def log_exit(self, face_id: str, image_path: str | None, dwell_time: int = 0):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"EVENT: EXIT | Face ID: {face_id} | Timestamp: {ts} | Dwell Time: {dwell_time}s | Image: {image_path or 'none'}")

    def log_registration(self, face_id: str, image_path: str | None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"EVENT: REGISTERED | Face ID: {face_id} | Timestamp: {ts} | Image: {image_path or 'none'}")

    def log_embedding_generated(self, face_id: str, similarity: float):
        self.logger.info(f"EVENT: EMBEDDING | Face ID: {face_id} | Best Sim: {similarity:.4f}")

    def log_tracking(self, track_id: int, face_id: str, state: str):
        self.logger.debug(f"EVENT: TRACKING | Track ID: {track_id} | Face ID: {face_id} | State: {state}")

    def log_frame_stats(self, frame_num: int, n_detections: int, n_tracks: int, unique_count: int):
        self.logger.debug(f"FRAME | #{frame_num} | detections={n_detections} | tracks={n_tracks} | unique={unique_count}")

