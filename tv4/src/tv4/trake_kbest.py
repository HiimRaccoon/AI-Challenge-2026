"""K-best temporal alignment for TRAKE candidate pools.

The implementation is deterministic, bounded, and operates only on existing
canonical hypotheses. It never derives frame IDs from timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import SearchCandidate
from .wp12_trake import EventPool


@dataclass(frozen=True)
class SequencePath:
    candidates: tuple[SearchCandidate, ...]
    score: float

    @property
    def identity(self) -> tuple[tuple[str, int], ...]:
        return tuple((candidate.video_id, candidate.frame_id) for candidate in self.candidates)


def _deduplicate_pool(candidates: Sequence[SearchCandidate], limit: int) -> tuple[SearchCandidate, ...]:
    best: dict[tuple[str, int], SearchCandidate] = {}
    for candidate in candidates:
        identity = (candidate.video_id, candidate.frame_id)
        current = best.get(identity)
        if current is None or (candidate.score or 0.0) > (current.score or 0.0):
            best[identity] = candidate
    ordered = sorted(
        best.values(),
        key=lambda candidate: (
            -(candidate.score or 0.0), candidate.timestamp_ms,
            candidate.frame_id, candidate.video_id,
        ),
    )
    return tuple(ordered[:limit])


def _top_unique(paths: Sequence[SequencePath], limit: int) -> list[SequencePath]:
    unique: dict[tuple[tuple[str, int], ...], SequencePath] = {}
    for path in paths:
        current = unique.get(path.identity)
        if current is None or path.score > current.score:
            unique[path.identity] = path
    return sorted(
        unique.values(),
        key=lambda path: (
            -path.score,
            tuple(candidate.timestamp_ms for candidate in path.candidates),
            path.identity,
        ),
    )[:limit]


def align_k_best(
    pools: Sequence[EventPool],
    *,
    k: int = 5,
    max_candidates_per_event: int = 30,
) -> list[SequencePath]:
    """Return up to ``k`` complete, same-video, strictly ordered paths."""
    if k < 1 or max_candidates_per_event < 1:
        raise ValueError("k and max_candidates_per_event must be positive")
    if not pools or any(not pool.candidates for pool in pools):
        return []

    trimmed = [_deduplicate_pool(pool.candidates, max_candidates_per_event) for pool in pools]
    if any(not candidates for candidates in trimmed):
        return []
    video_ids = {candidate.video_id for candidates in trimmed for candidate in candidates}
    if len(video_ids) != 1:
        raise ValueError("event pools must be scoped to exactly one video")

    states: list[list[SequencePath]] = [
        [SequencePath((candidate,), candidate.score or 0.0)]
        for candidate in trimmed[0]
    ]
    for event_candidates in trimmed[1:]:
        next_states: list[list[SequencePath]] = []
        for candidate in event_candidates:
            extensions: list[SequencePath] = []
            for previous_paths in states:
                for previous in previous_paths:
                    if previous.candidates[-1].timestamp_ms >= candidate.timestamp_ms:
                        continue
                    extensions.append(SequencePath(
                        candidates=(*previous.candidates, candidate),
                        score=previous.score + (candidate.score or 0.0),
                    ))
            next_states.append(_top_unique(extensions, k))
        states = next_states

    return _top_unique([path for paths in states for path in paths], k)


def event_alternatives(
    pools: Sequence[EventPool],
    aligned: Sequence[SearchCandidate],
    *,
    limit: int = 5,
) -> tuple[tuple[SearchCandidate, ...], ...]:
    """Return bounded alternatives that cannot violate locked-neighbor order."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(pools) != len(aligned):
        raise ValueError("pools and aligned sequence must have equal length")

    alternatives: list[tuple[SearchCandidate, ...]] = []
    for index, (pool, selected) in enumerate(zip(pools, aligned)):
        lower = aligned[index - 1].timestamp_ms if index > 0 else -1
        upper = aligned[index + 1].timestamp_ms if index + 1 < len(aligned) else None
        candidates = [
            candidate for candidate in _deduplicate_pool(pool.candidates, max(limit * 4, limit))
            if candidate.video_id == selected.video_id
            and candidate.frame_id != selected.frame_id
            and candidate.timestamp_ms > lower
            and (upper is None or candidate.timestamp_ms < upper)
        ]
        alternatives.append(tuple(candidates[:limit]))
    return tuple(alternatives)
