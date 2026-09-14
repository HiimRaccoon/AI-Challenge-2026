"""Stable data contracts shared by WP03 components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"


class ContractError(ValueError):
    """Raised when a pipeline record violates a WP03 contract."""


def utc_now_iso8601() -> str:
    """Return a compact, timezone-explicit UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class FrameRecord:
    preprocess_run_id: str
    video_id: str
    frame_id: int
    keyframe_seq: int
    timestamp_ms: int
    pts: int | None
    time_base: str | None
    decode_index: int | None
    shot_id: str
    keyframe_path: str

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "FrameRecord":
        pts = record.get("pts")
        decode_index = record.get("decode_index")
        if pts is not None:
            _require_int(pts, "pts", minimum=0)
        if decode_index is not None:
            _require_int(decode_index, "decode_index", minimum=0)
        time_base = record.get("time_base")
        if time_base is not None and not isinstance(time_base, str):
            raise ContractError("time_base must be a string or null")
        return cls(
            preprocess_run_id=_require_string(record.get("preprocess_run_id"), "preprocess_run_id"),
            video_id=_require_string(record.get("video_id"), "video_id"),
            frame_id=_require_int(record.get("frame_id"), "frame_id"),
            keyframe_seq=_require_int(record.get("keyframe_seq"), "keyframe_seq"),
            timestamp_ms=_require_int(record.get("timestamp_ms"), "timestamp_ms"),
            pts=pts,
            time_base=time_base,
            decode_index=decode_index,
            shot_id=_require_string(record.get("shot_id"), "shot_id"),
            keyframe_path=_require_string(record.get("keyframe_path"), "keyframe_path"),
        )


@dataclass(frozen=True)
class EmbeddingMapRecord:
    schema_version: str
    preprocess_run_id: str
    model_name: str
    model_version: str
    vector_id: int
    video_id: str
    frame_id: int
    keyframe_seq: int
    timestamp_ms: int
    shot_id: str
    embedding_dim: int
    vector_dtype: str
    l2_normalized: bool
    keyframe_path: str
    created_at_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "preprocess_run_id": self.preprocess_run_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "vector_id": self.vector_id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "keyframe_seq": self.keyframe_seq,
            "timestamp_ms": self.timestamp_ms,
            "shot_id": self.shot_id,
            "embedding_dim": self.embedding_dim,
            "vector_dtype": self.vector_dtype,
            "l2_normalized": self.l2_normalized,
            "keyframe_path": self.keyframe_path,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True)
class SearchRequest:
    query_id: str
    task: str
    query_text: str | None
    question: str | None
    events: Sequence[str]
    filters: Mapping[str, object]
    limit: int
    language: str | None
    session_id: str | None
    event_index: int | None = None
    internal_pool: bool = False

    def __post_init__(self) -> None:
        _require_string(self.query_id, "query_id")
        if self.task not in {"KIS", "VQA", "TRAKE"}:
            raise ContractError("task must be KIS, VQA, or TRAKE")
        max_limit = 1_000 if self.internal_pool else 100
        if self.limit < 1 or self.limit > max_limit:
            raise ContractError(f"limit must be between 1 and {max_limit}")
        if self.query_text is not None and not isinstance(self.query_text, str):
            raise ContractError("query_text must be a string or null")
        if self.question is not None and not isinstance(self.question, str):
            raise ContractError("question must be a string or null")
        if not all(isinstance(event, str) for event in self.events):
            raise ContractError("events must contain strings")
        if not isinstance(self.filters, Mapping):
            raise ContractError("filters must be an object")
        if self.language is not None and not isinstance(self.language, str):
            raise ContractError("language must be a string or null")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise ContractError("session_id must be a string or null")
        if self.event_index is not None:
            _require_int(self.event_index, "event_index")

    def visual_query_text(self) -> str:
        for value in (self.query_text, self.question):
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ContractError("query_text or question must be non-empty for visual retrieval")


@dataclass(frozen=True)
class SearchCandidate:
    schema_version: str
    query_id: str
    event_index: int | None
    video_id: str
    frame_id: int
    timestamp_ms: int
    source: str
    raw_score: float | None
    score: float | None
    rank: int
    model_scores: Mapping[str, float]
    model_ranks: Mapping[str, int]
    matched_filters: Sequence[str]
    evidence_refs: Sequence[str]
    confidence: float | None
    preprocess_run_id: str
    created_at_utc: str

    @classmethod
    def visual_rrf(
        cls,
        *,
        query_id: str,
        event_index: int | None,
        preprocess_run_id: str,
        video_id: str,
        frame_id: int,
        timestamp_ms: int,
        rank: int,
        rrf_score: float,
        model_scores: Mapping[str, float],
        model_ranks: Mapping[str, int],
        keyframe_path: str,
    ) -> "SearchCandidate":
        return cls(
            schema_version=SCHEMA_VERSION,
            query_id=query_id,
            event_index=event_index,
            video_id=video_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source="visual",
            raw_score=None,
            score=rrf_score,
            rank=rank,
            model_scores=dict(model_scores),
            model_ranks=dict(model_ranks),
            matched_filters=(),
            evidence_refs=(f"keyframe:{keyframe_path}",),
            confidence=None,
            preprocess_run_id=preprocess_run_id,
            created_at_utc=utc_now_iso8601(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "event_index": self.event_index,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "source": self.source,
            "raw_score": self.raw_score,
            "score": self.score,
            "rank": self.rank,
            "model_scores": dict(self.model_scores),
            "model_ranks": dict(self.model_ranks),
            "matched_filters": list(self.matched_filters),
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "preprocess_run_id": self.preprocess_run_id,
            "created_at_utc": self.created_at_utc,
        }


@dataclass(frozen=True)
class SearchResponse:
    schema_version: str
    query_id: str
    wp03_run_id: str
    preprocess_run_id: str
    degraded: bool
    models_requested: tuple[str, ...]
    models_used: tuple[str, ...]
    requested_top_k: int
    returned_count: int
    candidate_k_per_model: int
    hard_candidate_cap: int | None
    candidates: tuple[SearchCandidate, ...]

    @classmethod
    def create(
        cls,
        *,
        query_id: str,
        wp03_run_id: str,
        preprocess_run_id: str,
        requested_top_k: int,
        candidate_k_per_model: int,
        hard_candidate_cap: int | None,
        models_requested: Sequence[str],
        models_used: Sequence[str],
        candidates: Sequence[SearchCandidate],
    ) -> "SearchResponse":
        if requested_top_k <= 0:
            raise ContractError("requested_top_k must be positive")
        if candidate_k_per_model <= 0:
            raise ContractError("candidate_k_per_model must be positive")
        if hard_candidate_cap is not None and hard_candidate_cap <= 0:
            raise ContractError("hard_candidate_cap must be positive or null")
        candidate_tuple = tuple(candidates)
        return cls(
            schema_version=SCHEMA_VERSION,
            query_id=query_id,
            wp03_run_id=wp03_run_id,
            preprocess_run_id=preprocess_run_id,
            degraded=tuple(models_requested) != tuple(models_used),
            models_requested=tuple(models_requested),
            models_used=tuple(models_used),
            requested_top_k=requested_top_k,
            returned_count=len(candidate_tuple),
            candidate_k_per_model=candidate_k_per_model,
            hard_candidate_cap=hard_candidate_cap,
            candidates=candidate_tuple,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "wp03_run_id": self.wp03_run_id,
            "preprocess_run_id": self.preprocess_run_id,
            "degraded": self.degraded,
            "models_requested": list(self.models_requested),
            "models_used": list(self.models_used),
            "requested_top_k": self.requested_top_k,
            "returned_count": self.returned_count,
            "candidate_k_per_model": self.candidate_k_per_model,
            "hard_candidate_cap": self.hard_candidate_cap,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class ShardManifest:
    shard_id: int
    record_start: int
    record_end: int
    shard_input_digest: str
    output_sha256: str
    shape: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class ModelManifest:
    model_key: str
    status: str
    preprocess_run_id: str
    vector_count: int
