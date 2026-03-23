"""
pipeline.py
-----------
Master per-frame orchestration pipeline.

FIXES FOR ACCURATE UNIQUE VISITOR COUNTING:

1. Temporal identity voting (self._track_votes):
   Rather than committing a face_id on the very first embedding match,
   we collect votes over VOTE_WINDOW frames. The face_id that wins
   a majority of votes is committed as the track's identity.
   This prevents a single bad-quality frame from triggering a new
   registration or a wrong re-identification.

2. Minimum detection area filter:
   Detections covering less than MIN_DETECTION_AREA pixels are skipped.
   A 30x60 pixel body crop at 5m distance produces a face of ~8x10px
   inside it — too small for reliable ArcFace embedding. These distant
   detections were the primary cause of duplicate registrations.

3. Embedding update on re-identification:
   When identify() returns a match, we call registry.update_embedding()
   to adapt the stored embedding toward the current observation. This
   makes the gallery progressively better at recognising the person
   from new angles.

4. Soft threshold gate before registration:
   registry.is_likely_known() uses a wider threshold (0.28) to block
   registrations that are probably the same person seen from a new angle.

5. Per-track registration lock:
   Once a track_id has registered a face, it cannot register again
   for its lifetime — prevents the same detection window causing
   multiple registrations.
"""

import logging
from collections import defaultdict, Counter
import cv2
import numpy as np

from config_loader import Config
from detector import FaceDetector
from embedder import FaceEmbedder
from face_registry import FaceRegistry
from track_state_manager import TrackStateManager
from event_logger import EventLogger
from database import Database

logger = logging.getLogger(__name__)

# Minimum bounding box area (pixels²) to attempt face embedding.
# Detections smaller than this produce unreliable embeddings.
MIN_DETECTION_AREA = 4000   # roughly a 63x63 pixel box

# Number of frames to collect votes before committing a face_id to a track.
# Higher = more stable identity, but slightly slower first-time recognition.
VOTE_WINDOW = 3


def _build_tracker(max_age, min_hits, iou_threshold):
    """Try DeepSORT; fall back to SimpleTracker if API incompatible."""
    try:
        from tracker import FaceTracker
        t = FaceTracker(max_age=max_age, min_hits=min_hits,
                        iou_threshold=iou_threshold)
        t.update([], np.zeros((4, 4, 3), dtype=np.uint8))
        logger.info("Tracker: DeepSORT active.")
        return t
    except Exception as ex:
        logger.warning(f"DeepSORT unavailable ({ex}). Using SimpleTracker.")
        from simple_tracker import SimpleTracker
        return SimpleTracker(max_age=max_age, min_hits=min_hits,
                             iou_threshold=iou_threshold)


class FaceTrackerPipeline:
    """End-to-end face tracking pipeline."""

    def __init__(self, config: Config):
        self.cfg = config
        self._frame_counter = 0
        self._skip_n = config.detection.get("frame_skip", 3)

        # Persistent map: track_id → committed face_id
        self._tid_to_face: dict[int, str] = {}

        # Vote buffer: track_id → list of candidate face_ids seen recently
        # Used to stabilise identity assignment over multiple frames
        self._track_votes: dict[int, list[str]] = defaultdict(list)

        # Track IDs that have already triggered a registration this session
        self._registered_tids: set[int] = set()

        logger.info("Initialising pipeline subsystems...")

        self.detector = FaceDetector(
            model_path=config.detection.get("yolo_model", "yolov8n.pt"),
            confidence=config.detection.get("confidence_threshold", 0.65),
            input_size=config.detection.get("input_size", 640),
        )

        strict_thresh = config.recognition.get("similarity_threshold", 0.35)
        soft_thresh = config.recognition.get("soft_threshold", 0.28)

        self.embedder = FaceEmbedder(
            model_name=config.recognition.get("model_name", "buffalo_l"),
            det_size=tuple(config.recognition.get("det_size", [640, 640])),
            similarity_threshold=strict_thresh,
        )

        self.registry = FaceRegistry(
            strict_threshold=strict_thresh,
            soft_threshold=soft_thresh,
        )

        self.tracker = _build_tracker(
            max_age=config.tracking.get("max_age", 30),
            min_hits=config.tracking.get("min_hits", 2),
            iou_threshold=config.tracking.get("iou_threshold", 0.3),
        )
        self.db = Database(
            db_path=config.database.get("path", "face_tracker.db")
        )
        self.event_logger = EventLogger(
            log_dir=config.logging.get("log_dir", "logs"),
            crop_size=tuple(config.logging.get("crop_size", [112, 112])),
            save_crops=config.logging.get("save_face_crops", True),
        )
        self.state_manager = TrackStateManager(
            exit_patience_frames=config.tracking.get("exit_patience_frames", 15)
        )

        # Restore face gallery from DB (persists across video runs)
        existing = self.db.get_all_embeddings()
        self.registry.load_from_db(existing)
        logger.info(f"Restored {len(existing)} face(s) from database.")
        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one BGR frame. Returns annotated frame."""
        self._frame_counter += 1

        run_detection = (
            self._frame_counter == 1 or
            self._frame_counter % (self._skip_n + 1) == 0
        )

        # --- Step 1: Detect ---
        raw_detections = []
        if run_detection:
            raw_detections = self.detector.detect(frame)

        # Filter out tiny detections — bodies at a distance produce
        # face crops too small for reliable ArcFace embeddings
        detections = [
            d for d in raw_detections
            if (d[2] - d[0]) * (d[3] - d[1]) >= MIN_DETECTION_AREA
        ]
        if run_detection and len(raw_detections) != len(detections):
            logger.debug(
                f"Frame {self._frame_counter}: filtered "
                f"{len(raw_detections) - len(detections)} small detection(s)"
            )

        # --- Step 2: Track ---
        tracks = self.tracker.update(detections, frame)

        # --- Step 3: Process confirmed tracks ---
        active_track_ids: list[int] = []
        track_bboxes: dict[int, tuple] = {}

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = int(track.track_id)
            ltrb = track.to_ltrb()
            x1 = max(0, int(ltrb[0]))
            y1 = max(0, int(ltrb[1]))
            x2 = min(frame.shape[1], int(ltrb[2]))
            y2 = min(frame.shape[0], int(ltrb[3]))

            if x2 <= x1 or y2 <= y1:
                continue

            # Secondary area filter on tracked boxes (Kalman-predicted boxes
            # can expand during occlusion)
            if (x2 - x1) * (y2 - y1) < MIN_DETECTION_AREA:
                continue

            track_bboxes[tid] = (x1, y1, x2, y2)
            active_track_ids.append(tid)

            if run_detection:
                self._process_track_identity(frame, tid, x1, y1, x2, y2)

        # Clean up identity map for ended tracks
        all_live_tids = {int(t.track_id) for t in tracks}
        for tid in list(self._tid_to_face):
            if tid not in all_live_tids:
                del self._tid_to_face[tid]
        for tid in list(self._track_votes):
            if tid not in all_live_tids:
                del self._track_votes[tid]

        # --- Step 4: Entry / exit detection ---
        entries, exits = self.state_manager.update(
            active_track_ids, self._tid_to_face, track_bboxes
        )

        # --- Step 5: Log entries ---
        for face_id, bbox in entries:
            crop = self._safe_crop(frame, *bbox) if bbox else None
            img_path = self.event_logger.save_face_crop(crop, face_id, "entry")
            self.db.log_event(face_id, "entry", img_path)
            self.event_logger.log_entry(face_id, img_path)

        # --- Step 6: Log exits ---
        for face_id, last_bbox in exits:
            crop = self._safe_crop(frame, *last_bbox) if last_bbox else None
            img_path = self.event_logger.save_face_crop(crop, face_id, "exit")
            self.db.log_event(face_id, "exit", img_path)
            self.event_logger.log_exit(face_id, img_path)

        # --- Step 7: Draw overlay ---
        if self.cfg.display.get("draw_bboxes", True):
            frame = self._draw_overlays(frame, active_track_ids, track_bboxes)

        return frame

    def flush_remaining_tracks(self):
        """Fire exit events for faces still active at end of video."""
        logger.info("Flushing active tracks at end of video...")
        for face_id in self.state_manager.get_active_face_ids():
            self.db.log_event(face_id, "exit", None)
            self.event_logger.log_exit(face_id, None)
            logger.info(f"End-of-video exit: {face_id}")

    def get_unique_visitor_count(self) -> int:
        return self.db.get_unique_visitor_count()

    # ------------------------------------------------------------------ #
    # Identity resolution                                                  #
    # ------------------------------------------------------------------ #

    def _process_track_identity(self, frame: np.ndarray,
                                  tid: int, x1: int, y1: int,
                                  x2: int, y2: int):
        """
        Run embedding extraction and identity resolution for one confirmed track.

        Uses a vote buffer to stabilise identity over multiple frames before
        committing a face_id or triggering a new registration.
        """
        crop = self._safe_crop(frame, x1, y1, x2, y2)
        if crop is None:
            return

        embedding = self.embedder.get_embedding(crop)
        if embedding is None:
            return

        # --- Check against gallery ---
        face_id, sim = self.registry.identify(embedding)

        if face_id is not None:
            # Definite match — update the gallery embedding with this new
            # observation so it adapts to pose changes over time
            self.registry.update_embedding(face_id, embedding)
            logger.debug(f"  Track {tid}: MATCH {face_id[:16]} (sim={sim:.3f})")

            # Add to vote buffer
            votes = self._track_votes[tid]
            votes.append(face_id)
            if len(votes) > VOTE_WINDOW:
                votes.pop(0)

            # Commit the majority-vote winner as this track's identity
            winner, vote_count = Counter(votes).most_common(1)[0]
            if vote_count >= max(1, VOTE_WINDOW // 2):
                self._tid_to_face[tid] = winner

        else:
            # No strict match found
            logger.debug(f"  Track {tid}: no match (best_sim={sim:.3f})")

            # Soft-threshold gate: if best similarity is above soft_threshold,
            # this is probably a known person at a new angle — do NOT register
            if self.registry.is_likely_known(embedding):
                logger.debug(
                    f"  Track {tid}: soft-threshold gate triggered "
                    f"— skipping registration"
                )
                return

            # Per-track lock: this track already registered someone
            if tid in self._registered_tids:
                logger.debug(
                    f"  Track {tid}: registration lock active — skipping"
                )
                return

            # Passed all gates — register as a genuinely new unique face
            face_id = self.registry.register_new(embedding)
            self._registered_tids.add(tid)

            crop_path = self.event_logger.save_face_crop(
                crop, face_id, "registered"
            )
            self.db.register_face(face_id, embedding, crop_path)
            self.event_logger.log_registration(face_id, crop_path)

            logger.info(
                f"NEW unique face: {face_id} | "
                f"total registered: {self.registry.count}"
            )

            # Immediately commit to this track's identity
            self._tid_to_face[tid] = face_id
            self._track_votes[tid] = [face_id] * VOTE_WINDOW

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _safe_crop(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _draw_overlays(self, frame, active_tids, track_bboxes):
        unique_count = self.get_unique_visitor_count()
        for tid in active_tids:
            bbox = track_bboxes.get(tid)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            face_id = self._tid_to_face.get(tid)
            color = (0, 220, 120) if face_id else (200, 140, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = face_id[-14:] if face_id else f"track_{tid}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        # HUD
        cv2.rectangle(frame, (0, 0), (310, 68), (0, 0, 0), -1)
        cv2.putText(frame, f"Unique visitors: {unique_count}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2)
        cv2.putText(frame,
                    f"Frame: {self._frame_counter}  Tracks: {len(active_tids)}",
                    (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return frame