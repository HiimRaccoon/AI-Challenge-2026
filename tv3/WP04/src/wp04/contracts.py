from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


class ContractError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_box(box: tuple[float, float, float, float]) -> None:
    if len(box) != 4 or any(value < 0.0 or value > 1.0 for value in box):
        raise ContractError("bounding boxes must contain four normalized values")
    if box[0] > box[2] or box[1] > box[3]:
        raise ContractError("bounding box coordinates must be ordered")


@dataclass(frozen=True, slots=True)
class WP04RunIdentity:
    """Immutable provenance for one set of WP04 artifacts."""

    preprocess_run_id: str
    wp04_artifact_set_id: str
    input_fingerprint: str
    config_fingerprint: str


@dataclass(frozen=True, slots=True)
class AudioRecord:
    """TV1 audio declaration; absence is meaningful only when declared."""

    preprocess_run_id: str
    video_id: str
    audio_path: str | None
    checksum: str | None
    declared_present: bool
    duration_ms: int | None

    def __post_init__(self) -> None:
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ContractError("audio duration_ms must be non-negative")
        if self.declared_present and (not self.audio_path or not self.checksum):
            raise ContractError("declared audio must have a path and checksum")


@dataclass(frozen=True, slots=True)
class OCRDetection:
    preprocess_run_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    raw_text: str
    normalized_text: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    confidence: float
    model_name: str
    model_version: str
    evidence_id: str

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ContractError("OCR must preserve non-negative frame identity")
        _validate_box(self.bbox_xyxy_norm)
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("OCR confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess_run_id": self.preprocess_run_id, "video_id": self.video_id,
            "frame_id": self.frame_id, "timestamp_ms": self.timestamp_ms,
            "raw_text": self.raw_text, "normalized_text": self.normalized_text,
            "bbox_xyxy_norm": list(self.bbox_xyxy_norm), "confidence": self.confidence,
            "model_name": self.model_name, "model_version": self.model_version,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class ASRSegment:
    preprocess_run_id: str
    video_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    raw_text: str
    normalized_text: str
    confidence: float
    model_name: str
    model_version: str

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ContractError("ASR interval must be non-negative and ordered")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("ASR confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess_run_id": self.preprocess_run_id, "video_id": self.video_id,
            "segment_id": self.segment_id, "start_ms": self.start_ms, "end_ms": self.end_ms,
            "raw_text": self.raw_text, "normalized_text": self.normalized_text,
            "confidence": self.confidence, "model_name": self.model_name,
            "model_version": self.model_version,
        }


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    preprocess_run_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    label: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    confidence: float
    model_name: str
    model_version: str
    evidence_id: str

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ContractError("object must preserve non-negative frame identity")
        _validate_box(self.bbox_xyxy_norm)
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("object confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess_run_id": self.preprocess_run_id, "video_id": self.video_id,
            "frame_id": self.frame_id, "timestamp_ms": self.timestamp_ms,
            "label": self.label, "bbox_xyxy_norm": list(self.bbox_xyxy_norm),
            "confidence": self.confidence, "model_name": self.model_name,
            "model_version": self.model_version, "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    preprocess_run_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    fields: dict[str, Any]
    evidence_refs: tuple[str, ...]
    record_id: str

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ContractError("metadata must preserve non-negative frame identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess_run_id": self.preprocess_run_id, "video_id": self.video_id,
            "frame_id": self.frame_id, "timestamp_ms": self.timestamp_ms,
            "fields": self.fields, "evidence_refs": list(self.evidence_refs),
            "record_id": self.record_id,
        }


@dataclass(frozen=True, slots=True)
class FrameRecord:
    preprocess_run_id: str
    video_id: str
    frame_id: int
    keyframe_seq: int
    timestamp_ms: int
    keyframe_path: str | None = None
    keyframe_sha256: str | None = None
    shot_id: str | None = None
    selection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ContractError("frame_id and timestamp_ms must be non-negative")
        if (self.keyframe_path is None) != (self.keyframe_sha256 is None):
            raise ContractError("keyframe_path and keyframe_sha256 must be supplied together")


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query_id: str
    task: Literal["KIS", "VQA", "TRAKE"]
    query_text: str | None = None
    question: str | None = None
    events: tuple[str, ...] = ()
    filters: dict[str, Any] = field(default_factory=dict)
    limit: int = 100
    language: str | None = None
    session_id: str | None = None
    schema_version: str = "1.0.0"

    def validate(self) -> None:
        if not 1 <= self.limit <= 100:
            raise ContractError("limit must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    query_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    source: Literal["visual", "ocr", "asr", "metadata", "object", "feedback", "fusion"]
    rank: int
    preprocess_run_id: str
    schema_version: str = "1.0.0"
    event_index: int | None = None
    raw_score: float | None = None
    score: float | None = None
    model_scores: dict[str, float] = field(default_factory=dict)
    model_ranks: dict[str, int] = field(default_factory=dict)
    matched_filters: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    confidence: float | None = None
    created_at_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ContractError("rank must start at 1")
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ContractError("candidate must preserve non-negative original identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "query_id": self.query_id,
            "event_index": self.event_index, "video_id": self.video_id,
            "frame_id": self.frame_id, "timestamp_ms": self.timestamp_ms,
            "source": self.source, "raw_score": self.raw_score, "score": self.score,
            "rank": self.rank, "model_scores": self.model_scores,
            "model_ranks": self.model_ranks, "matched_filters": list(self.matched_filters),
            "evidence_refs": list(self.evidence_refs), "confidence": self.confidence,
            "preprocess_run_id": self.preprocess_run_id,
            "created_at_utc": self.created_at_utc,
        }
