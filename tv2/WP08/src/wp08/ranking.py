"""Model-local composed embedding math for the Top1 2025 baseline."""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping, Sequence
from .contracts import CandidateId


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("embedding must be a finite vector")
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("embedding must not have zero norm")
    return value / norm


def fuse_embedding(text_embedding: np.ndarray, image_embedding: np.ndarray, *, alpha: float = 0.75) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    text = _normalize(text_embedding)
    image = _normalize(image_embedding)
    if text.shape != image.shape:
        raise ValueError("text and image embeddings must share a shape")
    return _normalize(alpha * text + (1.0 - alpha) * image)


def late_rrf(rankings: Mapping[str, Sequence[CandidateId]], *, rrf_k: int = 60) -> tuple[CandidateId, ...]:
    """Fuse complete model-local rankings without mixing embedding spaces."""
    if rrf_k < 1 or len(rankings) != 4:
        raise ValueError("late RRF requires four rankings and a positive constant")
    expected = None
    scores: dict[CandidateId, float] = {}
    for candidates in rankings.values():
        sequence = tuple(candidates)
        if len(sequence) != len(set(sequence)):
            raise ValueError("model ranking contains duplicate candidates")
        identities = set(sequence)
        if expected is None:
            expected = identities
        elif identities != expected:
            raise ValueError("all four models must rank the same candidate pool")
        for rank, candidate in enumerate(sequence, 1):
            scores[candidate] = scores.get(candidate, 0.0) + 1.0 / (rrf_k + rank)
    return tuple(sorted(scores, key=lambda item: (-scores[item], item.video_id, item.frame_id)))
