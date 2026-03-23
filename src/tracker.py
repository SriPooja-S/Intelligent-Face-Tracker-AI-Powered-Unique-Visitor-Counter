"""
tracker.py
----------
DeepSORT wrapper that works with ANY installed version of deep_sort_realtime.

The deep_sort_realtime library has changed its API across versions:

  v1.2 and below:  embedder=None requires embedding as 4th tuple element
                   [([x,y,w,h], conf, cls, embedding), ...]

  v1.3.x:          embedder=None requires embeds= keyword argument
                   update_tracks(dets, embeds=[...], frame=frame)

  Some builds:     embedder=None requires embeddings= keyword argument
                   update_tracks(dets, embeddings=[...], frame=frame)

This file probes the API once at first call and picks the correct approach.
If all DeepSORT approaches fail, it falls back to the built-in SimpleTracker
so the pipeline always runs regardless of library version.
"""

import logging
import inspect
import numpy as np

logger = logging.getLogger(__name__)

# Will be set to one of: 'tuple4', 'embeds_kwarg', 'embeddings_kwarg', 'simple'
_API_MODE = None


def _detect_api_mode(tracker_instance) -> str:
    """
    Probe the installed DeepSort.update_tracks signature to find
    which calling convention it supports.
    """
    from deep_sort_realtime.deepsort_tracker import DeepSort
    try:
        sig = inspect.signature(DeepSort.update_tracks)
        params = list(sig.parameters.keys())
        logger.info(f"DeepSort.update_tracks params: {params}")

        if 'embeds' in params:
            return 'embeds_kwarg'
        elif 'embeddings' in params:
            return 'embeddings_kwarg'
        else:
            # No explicit embedding param — must be 4-tuple format
            return 'tuple4'
    except Exception as ex:
        logger.warning(f"Could not inspect DeepSort signature: {ex}")
        return 'tuple4'


class FaceTracker:
    """
    DeepSORT wrapper with automatic API version detection.
    Falls back to SimpleTracker if DeepSORT cannot be initialised.
    """

    EMBED_DIM = 128

    def __init__(self, max_age: int = 30,
                 min_hits: int = 2,
                 iou_threshold: float = 0.3):
        global _API_MODE

        self._use_simple = False

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._ds = DeepSort(
                max_age=max_age,
                n_init=min_hits,
                max_iou_distance=1.0 - iou_threshold,
                embedder=None,
                half=False,
            )
            # Detect which API mode to use
            _API_MODE = _detect_api_mode(self._ds)
            logger.info(
                f"DeepSORT initialised — mode={_API_MODE}, "
                f"max_age={max_age}, min_hits={min_hits}"
            )
            # Validate with a real test call
            self._test_call()
            logger.info("DeepSORT API test passed.")

        except Exception as ex:
            logger.warning(
                f"DeepSORT failed ({ex}). Switching to SimpleTracker."
            )
            self._use_simple = True
            from simple_tracker import SimpleTracker
            self._simple = SimpleTracker(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
            )

    def _test_call(self):
        """Run a single test frame to verify the API works end-to-end."""
        dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self._call_deepsort([(10, 10, 40, 40, 0.9)], dummy_frame)

    def _call_deepsort(self, detections: list[tuple],
                       frame: np.ndarray) -> list:
        """Call DeepSort using whichever API mode was detected."""
        global _API_MODE

        if not detections:
            return self._ds.update_tracks([], frame=frame)

        ds_dets = []
        for x1, y1, x2, y2, conf in detections:
            ds_dets.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], conf, "face"))

        dummy_embeds = [
            np.zeros(self.EMBED_DIM, dtype=np.float32)
            for _ in ds_dets
        ]

        if _API_MODE == 'embeds_kwarg':
            return self._ds.update_tracks(ds_dets, embeds=dummy_embeds, frame=frame)

        elif _API_MODE == 'embeddings_kwarg':
            return self._ds.update_tracks(ds_dets, embeddings=dummy_embeds, frame=frame)

        else:
            # 'tuple4' mode: embedding as 4th element of each detection tuple
            ds_dets_4 = [
                ([int(x1), int(y1), int(x2-x1), int(y2-y1)], conf, "face", emb)
                for (x1, y1, x2, y2, conf), emb in zip(detections, dummy_embeds)
            ]
            return self._ds.update_tracks(ds_dets_4, frame=frame)

    def update(self, detections: list[tuple],
               frame: np.ndarray) -> list:
        """
        Update tracker with the current frame's detections.

        Args:
            detections : List of (x1, y1, x2, y2, confidence) from YOLO.
            frame      : Current BGR frame.

        Returns:
            List of Track objects, each with:
              .track_id       — stable integer ID
              .to_ltrb()      — [x1, y1, x2, y2]
              .is_confirmed() — True once min_hits consecutive matches reached
        """
        if self._use_simple:
            return self._simple.update(detections, frame)

        try:
            return self._call_deepsort(detections, frame)
        except Exception as ex:
            logger.warning(
                f"DeepSORT update failed ({ex}). Falling back to SimpleTracker."
            )
            self._use_simple = True
            from simple_tracker import SimpleTracker
            self._simple = SimpleTracker()
            return self._simple.update(detections, frame)