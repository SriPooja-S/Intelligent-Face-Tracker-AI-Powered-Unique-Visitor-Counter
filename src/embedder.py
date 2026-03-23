"""
embedder.py
-----------
InsightFace ArcFace face embedding extractor.

FIX: Crops from YOLO (which detects full bodies) are often too large
for InsightFace to reliably find and align the face within them. This
version resizes the crop to a sensible range and rejects embeddings
from low-confidence detections so only reliable embeddings reach the
matching and registration stages.

The similarity_threshold is now only used by FaceRegistry — this
class just extracts and returns embeddings, keeping responsibilities
clearly separated.
"""

import logging
import numpy as np
import cv2
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Extracts 512-d L2-normalised ArcFace embeddings."""

    def __init__(self, model_name: str = "buffalo_l",
                 det_size: tuple = (640, 640),
                 similarity_threshold: float = 0.35):
        """
        Args:
            model_name           : InsightFace model pack name.
            det_size             : Internal detection resolution.
            similarity_threshold : Passed through for use by callers.
                                   Not used internally by this class.
        """
        self.threshold = similarity_threshold

        logger.info(f"Loading InsightFace model: {model_name}")
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        logger.info("InsightFace model loaded.")

    def get_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """
        Extract a 512-d L2-normalised embedding from a BGR image crop.

        The crop can be a full body bounding box from YOLO — InsightFace
        will run its own internal face detector on it to find and align
        the face region. Returns None if no face is found or quality
        is too low for a reliable embedding.
        """
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]

        # Minimum size check — too small means the face within the crop
        # will be just a few pixels, unusable for embedding
        if h < 40 or w < 40:
            return None

        # Ensure uint8 BGR
        if crop.dtype != np.uint8:
            crop = np.clip(crop * 255, 0, 255).astype(np.uint8)

        # Resize very large crops (full-body from close range) to speed up
        # InsightFace's internal detector while keeping enough resolution
        max_dim = max(h, w)
        if max_dim > 480:
            scale = 480.0 / max_dim
            crop = cv2.resize(
                crop,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        try:
            faces = self.app.get(crop)
        except Exception as ex:
            logger.debug(f"InsightFace error: {ex}")
            return None

        if not faces:
            return None

        # Use the highest-confidence detection
        best = max(faces, key=lambda f: f.det_score)

        # Reject low-confidence face detections — these produce noisy
        # embeddings that cause false new registrations
        if best.det_score < 0.50:
            logger.debug(
                f"Rejected low-confidence face: det_score={best.det_score:.3f}"
            )
            return None

        return best.normed_embedding.astype(np.float32)