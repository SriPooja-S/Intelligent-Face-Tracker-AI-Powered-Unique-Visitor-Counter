"""
track_state_manager.py
----------------------
Per-face entry/exit state machine.

Guarantees exactly ONE entry and ONE exit event per face per video session.

State transitions:
  (new track with face_id) → ACTIVE   → fires entry event once
  (track absent N frames)  → PENDING  → waits
  (absent > patience)      → EXITED   → fires exit event once
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class TrackStatus(Enum):
    ACTIVE = auto()
    PENDING_EXIT = auto()
    EXITED = auto()


@dataclass
class TrackRecord:
    face_id: str
    track_id: int
    status: TrackStatus = TrackStatus.ACTIVE
    entry_fired: bool = False
    exit_fired: bool = False
    frames_missing: int = 0
    last_bbox: tuple = field(default_factory=tuple)


class TrackStateManager:
    """Maintains track lifecycle and fires entry/exit events exactly once."""

    def __init__(self, exit_patience_frames: int = 15):
        self.exit_patience = exit_patience_frames
        # track_id → TrackRecord
        self._tracks: dict[int, TrackRecord] = {}

    def update(self,
               active_track_ids: list[int],
               track_to_face: dict[int, str],
               track_bboxes: dict[int, tuple]
               ) -> tuple[list[tuple], list[tuple]]:
        """
        Called once per frame.

        Returns:
            entries: [(face_id, bbox), ...] — new entrants this frame
            exits:   [(face_id, last_bbox), ...] — faces that just exited
        """
        entries: list[tuple] = []
        exits: list[tuple] = []
        active_set = set(active_track_ids)

        # Process tracks currently visible
        for tid in active_track_ids:
            face_id = track_to_face.get(tid)
            if face_id is None:
                continue  # Track confirmed but not yet matched to a face identity

            bbox = track_bboxes.get(tid, ())

            if tid not in self._tracks:
                # Brand-new confirmed track
                record = TrackRecord(face_id=face_id, track_id=tid, last_bbox=bbox)
                self._tracks[tid] = record
                logger.debug(f"New track record: tid={tid} face={face_id}")
            else:
                record = self._tracks[tid]
                record.status = TrackStatus.ACTIVE
                record.frames_missing = 0
                record.last_bbox = bbox
                # Handle re-id after brief occlusion
                if record.face_id != face_id:
                    logger.debug(f"Re-id: {record.face_id} → {face_id}")
                    record.face_id = face_id

            # Fire entry event exactly once
            if not record.entry_fired:
                record.entry_fired = True
                entries.append((face_id, bbox))
                logger.info(f"ENTRY queued: face_id={face_id}")

        # Process tracks NOT visible this frame
        for tid, record in list(self._tracks.items()):
            if tid in active_set or record.status == TrackStatus.EXITED:
                continue

            record.frames_missing += 1
            record.status = TrackStatus.PENDING_EXIT

            if record.frames_missing >= self.exit_patience:
                record.status = TrackStatus.EXITED
                if record.entry_fired and not record.exit_fired:
                    record.exit_fired = True
                    exits.append((record.face_id, record.last_bbox))
                    logger.info(f"EXIT queued: face_id={record.face_id}")

        return entries, exits

    def get_active_face_ids(self) -> list[str]:
        """Return face IDs for all tracks that haven't exited yet."""
        return [
            r.face_id for r in self._tracks.values()
            if r.status != TrackStatus.EXITED and r.entry_fired and not r.exit_fired
        ]

    def reset(self):
        """Clear all track state — call between video files if needed."""
        self._tracks.clear()
        logger.info("Track state manager reset.")
