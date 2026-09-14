"""Deterministic reciprocal-rank fusion for visual model results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .contracts import SearchCandidate


@dataclass(frozen=True)
class RankedHit:
    vector_id: int
    video_id: str
    frame_id: int
    timestamp_ms: int
    keyframe_path: str
    rank: int
    similarity: float


@dataclass(frozen=True)
class RrfResult:
    candidates: tuple[SearchCandidate, ...]


def diversify_visual_candidates(
    candidates: Sequence[SearchCandidate], *, limit: int, dedup_window_ms: int
) -> tuple[SearchCandidate, ...]:
    """Drop near-duplicate frames, then alternate videos without changing scores."""

    if limit <= 0:
        return ()
    if dedup_window_ms < 0:
        raise ValueError("dedup_window_ms must not be negative")
    kept_timestamps: dict[str, list[int]] = {}
    by_video: dict[str, list[SearchCandidate]] = {}
    video_order: list[str] = []
    for candidate in candidates:
        seen = kept_timestamps.setdefault(candidate.video_id, [])
        if any(abs(candidate.timestamp_ms - timestamp) <= dedup_window_ms for timestamp in seen):
            continue
        seen.append(candidate.timestamp_ms)
        if candidate.video_id not in by_video:
            by_video[candidate.video_id] = []
            video_order.append(candidate.video_id)
        by_video[candidate.video_id].append(candidate)
    selected: list[SearchCandidate] = []
    offsets = {video_id: 0 for video_id in video_order}
    while len(selected) < limit:
        progressed = False
        for video_id in video_order:
            offset = offsets[video_id]
            bucket = by_video[video_id]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[video_id] = offset + 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return tuple(replace(candidate, rank=rank) for rank, candidate in enumerate(selected, start=1))


def fuse_rrf(
    model_hits: Mapping[str, Sequence[RankedHit]],
    *,
    query_id: str,
    event_index: int | None,
    preprocess_run_id: str,
    limit: int,
    rrf_k: int = 60,
) -> RrfResult:
    """Fuse ranks by frame identity while keeping model scores separate."""

    aggregates: dict[tuple[str, int], dict[str, object]] = {}
    for model_key, hits in model_hits.items():
        for hit in hits:
            key = (hit.video_id, hit.frame_id)
            aggregate = aggregates.setdefault(
                key,
                {
                    "timestamp_ms": hit.timestamp_ms,
                    "keyframe_path": hit.keyframe_path,
                    "rrf_score": 0.0,
                    "model_scores": {},
                    "model_ranks": {},
                },
            )
            aggregate["rrf_score"] = float(aggregate["rrf_score"]) + 1.0 / (rrf_k + hit.rank)
            cast_scores = aggregate["model_scores"]
            cast_ranks = aggregate["model_ranks"]
            assert isinstance(cast_scores, dict) and isinstance(cast_ranks, dict)
            cast_scores[model_key] = hit.similarity
            cast_ranks[model_key] = hit.rank

    ordered = sorted(
        aggregates.items(),
        key=lambda item: (
            -float(item[1]["rrf_score"]),
            -len(item[1]["model_ranks"]),
            item[0][0],
            item[0][1],
        ),
    )
    candidates: list[SearchCandidate] = []
    for rank, ((video_id, frame_id), aggregate) in enumerate(ordered[:limit], start=1):
        candidates.append(
            SearchCandidate.visual_rrf(
                query_id=query_id,
                event_index=event_index,
                preprocess_run_id=preprocess_run_id,
                video_id=video_id,
                frame_id=frame_id,
                timestamp_ms=int(aggregate["timestamp_ms"]),
                rank=rank,
                rrf_score=float(aggregate["rrf_score"]),
                model_scores=aggregate["model_scores"],
                model_ranks=aggregate["model_ranks"],
                keyframe_path=str(aggregate["keyframe_path"]),
            )
        )
    return RrfResult(candidates=tuple(candidates))
