"""
simple_tracker.py
-----------------
Pure Python/NumPy IoU-based multi-object tracker.
Zero external dependencies beyond numpy.

Used as automatic fallback when DeepSORT's API is incompatible.
Provides identical interface to FaceTracker so pipeline.py works
without any changes regardless of which tracker is active.

Algorithm:
  - Each track has a bounding box, a stable ID, and age counters.
  - New detections are matched to existing tracks by IoU.
  - Unmatched detections start new tentative tracks.
  - Tracks confirmed after min_hits consecutive matches.
  - Tracks deleted after max_age frames with no match.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union for two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(area_a + area_b - inter)


class _Track:
    """Single track — same interface as deep_sort_realtime Track objects."""
    _id_counter = 1

    def __init__(self, bbox: tuple, min_hits: int):
        self.track_id = _Track._id_counter
        _Track._id_counter += 1
        self._bbox = bbox       # stored as (x1,y1,x2,y2)
        self._hits = 1
        self._age = 0
        self._min_hits = min_hits

    def update(self, bbox: tuple):
        self._bbox = bbox
        self._hits += 1
        self._age = 0

    def mark_missed(self):
        self._age += 1

    def is_confirmed(self) -> bool:
        return self._hits >= self._min_hits

    def is_stale(self, max_age: int) -> bool:
        return self._age >= max_age

    def to_ltrb(self) -> list:
        return list(self._bbox)


class SimpleTracker:
    """IoU tracker with the same .update() interface as FaceTracker."""

    def __init__(self, max_age: int = 30,
                 min_hits: int = 2,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_Track] = []
        logger.info(
            f"SimpleTracker initialised — "
            f"max_age={max_age}, min_hits={min_hits}, iou={iou_threshold}"
        )

    def update(self, detections: list[tuple],
               frame: np.ndarray) -> list:
        """
        Match detections to tracks and return all active tracks.
        Frame argument is unused but kept for API compatibility.
        """
        det_boxes = [
            (int(x1), int(y1), int(x2), int(y2))
            for x1, y1, x2, y2, *_ in detections
        ]

        matched_t = set()
        matched_d = set()

        # Build IoU matrix and match greedily
        if self._tracks and det_boxes:
            n_t, n_d = len(self._tracks), len(det_boxes)
            iou_mat = np.zeros((n_t, n_d), dtype=np.float32)
            for ti, t in enumerate(self._tracks):
                tb = tuple(t.to_ltrb())
                for di, db in enumerate(det_boxes):
                    iou_mat[ti, di] = _iou(tb, db)

            # Greedy: pick highest IoU pair repeatedly
            for _ in range(min(n_t, n_d)):
                if iou_mat.max() < self.iou_threshold:
                    break
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                matched_t.add(int(ti))
                matched_d.add(int(di))
                self._tracks[int(ti)].update(det_boxes[int(di)])
                iou_mat[int(ti), :] = 0
                iou_mat[:, int(di)] = 0

        # Age unmatched tracks
        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.mark_missed()

        # Create new tracks for unmatched detections
        for di, box in enumerate(det_boxes):
            if di not in matched_d:
                self._tracks.append(_Track(box, self.min_hits))

        # Remove stale tracks
        self._tracks = [t for t in self._tracks if not t.is_stale(self.max_age)]

        return list(self._tracks)