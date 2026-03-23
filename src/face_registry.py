"""
face_registry.py
----------------
In-memory gallery of registered face embeddings.

CORE FIX FOR ACCURATE UNIQUE COUNTING:

Problem: A single stored embedding per face is insufficient.
  The same person seen from a different angle, distance, or lighting
  produces embeddings that differ by 0.10-0.25 cosine distance.
  With only ONE stored embedding (the first one captured), later
  appearances at different angles fall below the match threshold
  and get registered as new people.

Solution — Multi-embedding gallery with running mean update:
  1. Each face stores up to MAX_EMBEDDINGS_PER_FACE embeddings.
  2. When a face is re-identified (strict match), its stored embedding
     is updated with a running mean: new_emb = 0.7*old + 0.3*current
     This adapts to pose/lighting changes over time.
  3. Matching checks against ALL stored embeddings for a face, taking
     the maximum similarity score. This handles angle variation.
  4. Soft dedup threshold prevents near-miss new registrations.
  5. Per-track lock prevents one track from registering twice.
"""

import logging
import uuid
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class FaceRecord:
    """
    Stores one or more embeddings for a single unique face.
    Uses running mean to adapt to pose/lighting changes.
    """

    # Maximum embeddings stored per face
    # More = better recall across angles, slightly slower matching
    MAX_EMBEDDINGS = 5

    # Weight for new embedding when updating the running mean
    # 0.3 = slowly adapt (stable), 0.5 = adapt faster
    UPDATE_WEIGHT = 0.3

    def __init__(self, face_id: str, embedding: np.ndarray):
        self.face_id = face_id
        # Store multiple embeddings for this face
        self._embeddings: list[np.ndarray] = [embedding.copy()]
        self.match_count = 1

    def best_similarity(self, query: np.ndarray) -> float:
        """
        Return the highest cosine similarity between the query
        and ANY of this face's stored embeddings.
        """
        return max(
            float(np.clip(np.dot(query, emb), -1.0, 1.0))
            for emb in self._embeddings
        )

    def update(self, new_embedding: np.ndarray):
        """
        Update the primary embedding using a running mean and
        optionally add a new embedding slot for a very different pose.
        """
        self.match_count += 1

        # Update the first (primary) embedding with running mean
        updated = (
            (1 - self.UPDATE_WEIGHT) * self._embeddings[0]
            + self.UPDATE_WEIGHT * new_embedding
        )
        # Re-normalise to unit vector (required for cosine similarity)
        norm = np.linalg.norm(updated)
        if norm > 1e-6:
            updated = updated / norm
        self._embeddings[0] = updated

        # If the new embedding is notably different from all stored ones,
        # add it as a separate entry to cover this new angle/pose
        if len(self._embeddings) < self.MAX_EMBEDDINGS:
            min_sim = min(
                float(np.dot(new_embedding, emb))
                for emb in self._embeddings
            )
            # Only add if meaningfully different (covers a new viewing angle)
            if min_sim < 0.70:
                self._embeddings.append(new_embedding.copy())

    @property
    def primary_embedding(self) -> np.ndarray:
        return self._embeddings[0]


class FaceRegistry:
    """
    In-memory multi-embedding gallery for fast per-frame face lookup.

    Matching strategy:
      - For each query embedding, check against ALL stored embeddings
        for ALL registered faces, taking the max similarity per face.
      - If max similarity >= strict_threshold: it's a known face.
      - If max similarity >= soft_threshold but < strict_threshold:
        it's probably a known face seen from a new angle — DO NOT register.
      - If max similarity < soft_threshold: register as genuinely new.
    """

    def __init__(self,
                 strict_threshold: float = 0.35,
                 soft_threshold: float = 0.28):
        """
        Args:
            strict_threshold : Similarity above which we call it a definite match.
            soft_threshold   : Similarity above which we suspect it's a known face
                               but just not matching well (block registration).
        """
        self.strict_threshold = strict_threshold
        self.soft_threshold = soft_threshold
        # face_id → FaceRecord
        self._records: dict[str, FaceRecord] = {}

    def load_from_db(self, db_pairs: list[tuple[str, np.ndarray]]):
        """
        Restore gallery from database at startup.
        Called once during pipeline initialisation.
        """
        for face_id, embedding in db_pairs:
            self._records[face_id] = FaceRecord(face_id, embedding)
        logger.info(
            f"Gallery loaded: {len(self._records)} face(s) restored from DB."
        )

    def identify(self, query: np.ndarray) -> tuple[str | None, float]:
        """
        Match query against all faces in the gallery.

        Returns:
            (face_id, similarity) if best match >= strict_threshold
            (None, best_similarity) otherwise
        """
        if not self._records:
            return None, 0.0

        best_id, best_sim = None, -1.0
        for face_id, record in self._records.items():
            sim = record.best_similarity(query)
            if sim > best_sim:
                best_sim = sim
                best_id = face_id

        if best_sim >= self.strict_threshold:
            return best_id, best_sim
        return None, best_sim

    def update_embedding(self, face_id: str, new_embedding: np.ndarray):
        """
        Update a known face's stored embeddings with a new observation.
        Call this when identify() returns a match to keep the gallery
        adapting to pose and lighting changes.
        """
        if face_id in self._records:
            self._records[face_id].update(new_embedding)

    def is_likely_known(self, query: np.ndarray) -> bool:
        """
        Check if a query is probably a known face using the soft threshold.
        Returns True if ANY stored face scores above soft_threshold.
        Use this as a registration gate to prevent near-miss duplicates.
        """
        if not self._records:
            return False
        _, best_sim = self.identify.__wrapped__(self, query) \
            if hasattr(self.identify, '__wrapped__') else (None, -1.0)
        # Recompute with soft threshold
        best_sim = max(
            (record.best_similarity(query) for record in self._records.values()),
            default=-1.0
        )
        return best_sim >= self.soft_threshold

    def register_new(self, embedding: np.ndarray) -> str:
        """
        Create a new face entry. Returns the new face_id.
        Caller is responsible for persisting to DB.
        """
        face_id = (
            f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
        self._records[face_id] = FaceRecord(face_id, embedding)
        logger.info(
            f"New face: {face_id} | gallery size: {len(self._records)}"
        )
        return face_id

    @property
    def count(self) -> int:
        return len(self._records)

    def get_all_face_ids(self) -> list[str]:
        return list(self._records.keys())