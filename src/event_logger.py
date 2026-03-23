"""
event_logger.py
---------------
Handles all file-system logging for the Face Tracker:

  1. Saves cropped face images to logs/{entries|exits|registered}/YYYY-MM-DD/
  2. Writes structured lines to logs/events.log for every critical system event.

Log line format (CSV-ish for easy parsing):
  [TIMESTAMP] [LEVEL] [MODULE] message
"""

import logging
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path


# ------------------------------------------------------------------ #
# File handler setup — called once from main.py                       #
# ------------------------------------------------------------------ #

def setup_logging(log_file: str = "logs/events.log",
                  log_level: str = "INFO"):
    """
    Configure the root logger to write to both console and events.log.
    Call this once at application startup before importing other modules.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# ------------------------------------------------------------------ #
# EventLogger class                                                    #
# ------------------------------------------------------------------ #

class EventLogger:
    """Saves face crop images and emits structured log entries."""

    def __init__(self, log_dir: str = "logs",
                 crop_size: tuple = (112, 112),
                 save_crops: bool = True):
        self.log_dir = Path(log_dir)
        self.crop_size = crop_size
        self.save_crops = save_crops
        self.logger = logging.getLogger("EventLogger")

    # ---------------------------------------------------------------- #
    # Image storage                                                     #
    # ---------------------------------------------------------------- #

    def _dated_dir(self, subfolder: str) -> Path:
        """Return logs/<subfolder>/YYYY-MM-DD/, creating it if needed."""
        today = datetime.now().strftime("%Y-%m-%d")
        path = self.log_dir / subfolder / today
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_face_crop(self, face_crop: np.ndarray,
                       face_id: str,
                       event_type: str) -> str | None:
        """
        Save a face crop to disk.

        Args:
            face_crop  : BGR numpy array.
            face_id    : The face's unique identifier string.
            event_type : 'entry', 'exit', or 'registered'.

        Returns:
            Absolute path to the saved image, or None if saving is disabled.
        """
        if not self.save_crops or face_crop is None or face_crop.size == 0:
            return None

        subdir_map = {
            "entry": "entries",
            "exit": "exits",
            "registered": "registered"
        }
        subdir = subdir_map.get(event_type, "entries")
        directory = self._dated_dir(subdir)

        ts = datetime.now().strftime("%H%M%S_%f")[:10]  # HHMMSSmmm
        filename = f"{face_id}_{ts}.jpg"
        filepath = directory / filename

        crop_resized = cv2.resize(face_crop, self.crop_size)
        cv2.imwrite(str(filepath), crop_resized)
        return str(filepath)

    # ---------------------------------------------------------------- #
    # Structured event logging                                          #
    # ---------------------------------------------------------------- #

    def log_entry(self, face_id: str, image_path: str | None):
        self.logger.info(
            f"ENTRY | face_id={face_id} | image={image_path or 'none'}"
        )

    def log_exit(self, face_id: str, image_path: str | None):
        self.logger.info(
            f"EXIT  | face_id={face_id} | image={image_path or 'none'}"
        )

    def log_registration(self, face_id: str, image_path: str | None):
        self.logger.info(
            f"REGISTERED | face_id={face_id} | image={image_path or 'none'}"
        )

    def log_embedding_generated(self, face_id: str, similarity: float):
        self.logger.debug(
            f"EMBEDDING | face_id={face_id} | best_sim={similarity:.4f}"
        )

    def log_tracking(self, track_id: int, face_id: str, state: str):
        self.logger.debug(
            f"TRACK  | track_id={track_id} | face_id={face_id} | state={state}"
        )

    def log_frame_stats(self, frame_num: int, n_detections: int,
                        n_tracks: int, unique_count: int):
        self.logger.debug(
            f"FRAME  | #{frame_num} | detections={n_detections} "
            f"| tracks={n_tracks} | unique={unique_count}"
        )
