"""TV4 shared data contracts.

Deliberately mirrors the SearchRequest / SearchCandidate contract already
defined by TV1/TV3 (see TV1_TV3_WP04/src/aic2026/contracts.py) and by TV2's
WP03 (see WP03/src/wp03/contracts.py) so JSON produced by any of the three
upstream services can be loaded here without translation, and so TV4's own
output can be consumed by TV5 unchanged.

Only the standard library is used here on purpose: TV4 must be importable
even in a bare CPU environment before any heavy retrieval dependency is
installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import re
from typing import Any, Literal, Mapping, Sequence

Task = Literal["KIS", "VQA", "TRAKE"]
Source = Literal["visual", "ocr", "asr", "metadata", "object", "feedback", "fusion"]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ContractError(ValueError):
    """Raised when a payload does not satisfy the TV4 contract."""


def exact_neighbor_response_is_safe(
    payload: object, video_id: str, anchor_frame_id: int, offsets: Sequence[int], preprocess_run_id: str,
    *, certification_id: str | None = None, certification_report_sha256: str | None = None,
    source_sha256: str | None = None, time_base: str | None = None,
) -> bool:
    """Reject malformed, fixture, or unproven WP09 subprocess payloads.

    This validation is intentionally repeated in TV4 instead of trusting a
    subprocess boundary.  A valid degraded response is allowed; any response
    that claims a selectable frame must provide every live-proof field.
    """
    if not isinstance(payload, Mapping) or payload.get("provenance_mode") != "live":
        return False
    if payload.get("video_id") != video_id or payload.get("anchor_frame_id") != anchor_frame_id:
        return False
    steps = payload.get("steps")
    if not isinstance(steps, list) or [step.get("offset") if isinstance(step, Mapping) else None for step in steps] != list(offsets):
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            return False
        frame = step.get("frame")
        if frame is None:
            continue
        if not isinstance(frame, Mapping):
            return False
        selectable = frame.get("submission_selectable") is True or frame.get("mapping_guaranteed") is True
        if not selectable:
            continue
        required = {
            "video_id": video_id,
            "preprocess_run_id": preprocess_run_id,
            "mapping_guaranteed": True,
            "submission_selectable": True,
            "identity_source": "certified_run_consecutive_original_decode",
            "media_identity_verified": True,
            "producer_compatibility_verified": True,
        }
        if not isinstance(frame.get("certification_id"), str) or not frame["certification_id"]:
            return False
        report_hash = frame.get("certification_report_sha256")
        if not isinstance(report_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", report_hash):
            return False
        if certification_id is not None and frame["certification_id"] != certification_id:
            return False
        if certification_report_sha256 is not None and report_hash != certification_report_sha256:
            return False
        if source_sha256 is not None and frame.get("source_sha256") != source_sha256:
            return False
        if time_base and frame.get("time_base") != time_base:
            return False
        if any(frame.get(name) != expected for name, expected in required.items()):
            return False
        if any(isinstance(frame.get(name), bool) or not isinstance(frame.get(name), int) or frame.get(name) < 0 for name in ("frame_id", "timestamp_ms", "pts")):
            return False
        if not isinstance(frame.get("time_base"), str) or not frame["time_base"]:
            return False
    return True


@dataclass(frozen=True)
class SearchRequest:
    query_id: str
    task: Task
    query_text: str | None = None
    question: str | None = None
    events: Sequence[str] = field(default_factory=tuple)
    filters: Mapping[str, Any] = field(default_factory=dict)
    limit: int = 100
    language: str | None = "vi"
    session_id: str | None = None
    event_index: int | None = None

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ContractError("query_id is required")
        if self.task not in ("KIS", "VQA", "TRAKE"):
            raise ContractError("task must be KIS, VQA or TRAKE")
        if not (1 <= self.limit <= 100):
            raise ContractError("limit must be in [1, 100]")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "query_id": self.query_id,
            "task": self.task,
            "query_text": self.query_text,
            "question": self.question,
            "events": list(self.events),
            "filters": dict(self.filters),
            "limit": self.limit,
            "language": self.language,
            "session_id": self.session_id,
            "event_index": self.event_index,
        }


@dataclass(frozen=True)
class SearchCandidate:
    query_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    source: Source
    rank: int
    schema_version: str = "1.0.0"
    event_index: int | None = None
    representative_frame_id: int | None = None
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    raw_score: float | None = None
    score: float | None = None
    model_scores: Mapping[str, float] = field(default_factory=dict)
    model_ranks: Mapping[str, int] = field(default_factory=dict)
    matched_filters: Sequence[str] = field(default_factory=tuple)
    evidence_refs: Sequence[str] = field(default_factory=tuple)
    provenance_sources: Sequence[str] = field(default_factory=tuple)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    preprocess_run_id: str = "unknown"
    created_at_utc: str = field(default_factory=now_utc)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SearchCandidate":
        known = {f: payload[f] for f in _CANDIDATE_FIELDS if f in payload}
        return cls(**known)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d


_CANDIDATE_FIELDS = {
    "query_id", "video_id", "frame_id", "timestamp_ms", "source", "rank",
    "schema_version", "event_index", "representative_frame_id",
    "window_start_ms", "window_end_ms", "raw_score", "score",
    "model_scores", "model_ranks", "matched_filters", "evidence_refs",
    "provenance_sources", "provenance", "confidence", "preprocess_run_id",
    "created_at_utc",
}


@dataclass(frozen=True)
class CanonicalFrameReference:
    """An upstream-provided original-frame identity used as VQA evidence."""

    video_id: str
    frame_id: int
    timestamp_ms: int
    keyframe_path: str | None = None
    preprocess_run_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    submission_selectable: bool = False

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ContractError("selected evidence frame requires video_id")
        for name, value in (("frame_id", self.frame_id), ("timestamp_ms", self.timestamp_ms)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"selected evidence frame {name} must be a non-negative integer")


def _normalized_bbox(value: object, *, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ContractError(f"{field_name} must be four normalized coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ContractError(f"{field_name} must be numeric")
    bbox = tuple(float(item) for item in value)
    x1, y1, x2, y2 = bbox
    if not all(0.0 <= item <= 1.0 for item in bbox) or x1 > x2 or y1 > y2:
        raise ContractError(f"{field_name} is not a normalized xyxy box")
    return bbox


def _canonical_non_negative(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class OcrEvidence:
    detection_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    raw_text: str
    normalized_text: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    polygon_norm: Sequence[Sequence[float]] | None = None
    confidence: float | None = None
    crop_evidence_path: str | None = None
    crop_sha256: str | None = None
    source_keyframe_sha256: str | None = None
    preprocess_run_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _canonical_non_negative(self.frame_id, field_name="OCR frame_id")
        _canonical_non_negative(self.timestamp_ms, field_name="OCR timestamp_ms")
        _normalized_bbox(self.bbox_xyxy_norm, field_name="OCR bbox_xyxy_norm")


@dataclass(frozen=True)
class AsrEvidence:
    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str
    normalized_text: str | None = None
    words: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    context: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    confidence: float | None = None
    language: str | None = None
    preprocess_run_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (("start_ms", self.start_ms), ("end_ms", self.end_ms)):
            _canonical_non_negative(value, field_name=f"ASR {name}")
        if self.start_ms > self.end_ms:
            raise ContractError("ASR start_ms must not exceed end_ms")


@dataclass(frozen=True)
class ObjectEvidence:
    detection_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    label: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    canonical_label: str | None = None
    confidence: float | None = None
    source_keyframe_path: str | None = None
    source_keyframe_sha256: str | None = None
    preprocess_run_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _canonical_non_negative(self.frame_id, field_name="Object frame_id")
        _canonical_non_negative(self.timestamp_ms, field_name="Object timestamp_ms")
        _normalized_bbox(self.bbox_xyxy_norm, field_name="Object bbox_xyxy_norm")


@dataclass(frozen=True)
class MetadataEvidence:
    metadata_id: str
    video_id: str
    source: str
    values: Mapping[str, Any]
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    confidence: float | None = None
    preprocess_run_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    source_record_sha256: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window_start_ms is not None and self.window_end_ms is not None and self.window_start_ms > self.window_end_ms:
            raise ContractError("metadata window_start_ms must not exceed window_end_ms")


@dataclass(frozen=True)
class EvidencePack:
    """Rich, provenance-preserving WP11 evidence for an advisory VQA proposal.

    The legacy text/label fields remain only for compatibility with existing
    answer engines.  They are never the sole representation when catalogue
    records are available.
    """

    query_id: str
    video_id: str
    frame_id: int
    timestamp_ms: int
    keyframe_path: str | None
    query_text: str | None = None
    question: str | None = None
    selected_frames: Sequence[CanonicalFrameReference] = field(default_factory=tuple)
    ocr_evidence: Sequence[OcrEvidence] = field(default_factory=tuple)
    asr_evidence: Sequence[AsrEvidence] = field(default_factory=tuple)
    object_evidence: Sequence[ObjectEvidence] = field(default_factory=tuple)
    metadata_evidence: Sequence[MetadataEvidence] = field(default_factory=tuple)
    availability: Mapping[str, str] = field(default_factory=dict)
    ocr_texts: Sequence[str] = field(default_factory=tuple)
    asr_texts: Sequence[str] = field(default_factory=tuple)
    object_labels: Sequence[str] = field(default_factory=tuple)
    neighbor_frame_ids: Sequence[int] = field(default_factory=tuple)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_usable_evidence(self) -> bool:
        return bool(
            self.selected_frames or self.ocr_evidence or self.asr_evidence
            or self.object_evidence or self.metadata_evidence
            or self.ocr_texts or self.asr_texts or self.object_labels
        )

    @property
    def has_malformed_evidence(self) -> bool:
        return any(status == "malformed" for status in self.availability.values())

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrakeEvent:
    event_index: int
    text: str


@dataclass(frozen=True)
class TrakeHypothesis:
    query_id: str
    video_id: str
    frame_ids: Sequence[int]
    event_scores: Sequence[float]
    aggregate_score: float
    preprocess_run_id: str
    timestamps_ms: Sequence[int] = ()
    candidates: Sequence[SearchCandidate] = ()
    rank: int = 1
    event_alternatives: Sequence[Sequence[SearchCandidate]] = ()
    verification: Mapping[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class FeedbackStartRequest:
    session_id: str
    original_query: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ContractError("session_id must be non-empty")
        if not self.original_query or not self.original_query.strip():
            raise ContractError("original_query must be non-empty")


@dataclass(frozen=True)
class FeedbackRefineRequest:
    session_id: str
    video_id: str
    frame_id: int
    feedback_text: str
    expected_revision: int
    source_candidate_frame_id: int | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ContractError("session_id must be non-empty")
        if not self.video_id or not self.video_id.strip():
            raise ContractError("video_id must be non-empty")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ContractError("frame_id must be non-negative integer")
        if self.source_candidate_frame_id is not None:
            if isinstance(self.source_candidate_frame_id, bool) or not isinstance(self.source_candidate_frame_id, int) or self.source_candidate_frame_id < 0:
                raise ContractError("source_candidate_frame_id must be non-negative integer")
        if not self.feedback_text or not self.feedback_text.strip():
            raise ContractError("feedback_text must be non-empty")
        if len(self.feedback_text) > 300:
            raise ContractError("feedback_text must not exceed 300 characters")
        if isinstance(self.expected_revision, bool) or not isinstance(self.expected_revision, int) or self.expected_revision < 0:
            raise ContractError("expected_revision must be non-negative integer")


@dataclass(frozen=True)
class FeedbackUndoRequest:
    session_id: str
    expected_revision: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ContractError("session_id must be non-empty")
        if isinstance(self.expected_revision, bool) or not isinstance(self.expected_revision, int) or self.expected_revision < 0:
            raise ContractError("expected_revision must be non-negative integer")


@dataclass(frozen=True)
class FeedbackResetRequest:
    session_id: str
    expected_revision: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ContractError("session_id must be non-empty")
        if isinstance(self.expected_revision, bool) or not isinstance(self.expected_revision, int) or self.expected_revision < 0:
            raise ContractError("expected_revision must be non-negative integer")
