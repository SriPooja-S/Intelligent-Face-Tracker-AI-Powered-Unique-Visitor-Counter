"""
detector.py
-----------
YOLOv8 face/person detector.

KEY FIX: Added automatic fallback — if the configured model file
doesn't exist, falls back to yolov8n.pt which ultralytics downloads
automatically. This prevents the FileNotFoundError on first run.
"""

import os
import logging
import numpy as np
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wraps a YOLOv8 model for face/person detection."""

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.5,
                 input_size: int = 640):

        self.confidence = confidence
        self.input_size = input_size

        # Fallback: if the specified model doesn't exist, use yolov8n.pt
        if not os.path.exists(model_path):
            logger.warning(
                f"Model '{model_path}' not found locally. "
                f"Falling back to 'yolov8n.pt' (will auto-download if needed)."
            )
            model_path = "yolov8n.pt"

        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded.")

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect faces/persons in a BGR frame.

        Returns list of (x1, y1, x2, y2, confidence).
        Filters to class 0 (person) for general YOLOv8 models,
        or all classes for face-specific models.
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.confidence,
            verbose=False,
            classes=[0]  # class 0 = person in COCO; all classes for face models
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Clamp to frame boundaries
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Skip tiny boxes — too small for reliable embedding
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    detections.append((x1, y1, x2, y2, conf))

        return detections

    def crop_face(self, frame: np.ndarray, bbox: tuple,
                  target_size: tuple = (112, 112)) -> np.ndarray:
        """Crop and resize a face region."""
        x1, y1, x2, y2 = bbox[:4]
        face = frame[max(0,y1):y2, max(0,x1):x2]
        if face.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)
        return cv2.resize(face, target_size)
