"""Immutable public contracts for original-video exact-frame refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence


class ContractError(ValueError):
    """Raised when a WP09 integration invariant is violated."""


class RefinementUnavailable(RuntimeError):
    """Original media/mapping cannot be used; callers must not fabricate a result."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class Task(StrEnum):
    KIS = "KIS"
    VQA = "VQA"
    TRAKE = "TRAKE"


class RefinementPolicy(StrEnum):
    REPRESENTATIVE = "representative"
    EVIDENCE_VISIBLE = "evidence_visible"
    TRANSITION = "transition"
    PEAK = "peak"
    STABLE_STATE = "stable_state"


class RefinementStatus(StrEnum):
    REFINED = "refined"
    PARTIAL = "partial"
    MANUAL_ONLY = "manual_only"


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _non_negative(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class RefinementContext:
    preprocess_run_id: str
    media_record_ref: str
    mapping_ref: str
    decoder_version: str
    model_version: str
    config_version: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _text(getattr(self, field), field)


@dataclass(frozen=True)
class CoarseCandidate:
    video_id: str
    frame_id: int
    timestamp_ms: int
    upstream_score: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _text(self.video_id, "video_id")
        _non_negative(self.frame_id, "frame_id")
        _non_negative(self.timestamp_ms, "timestamp_ms")
        for field in ("upstream_score", "confidence"):
            value = getattr(self, field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ContractError(f"{field} must be a number or null")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ContractError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class DecodeBudget:
    max_decoded_frames: int
    max_window_ms: int
    max_decode_time_ms: int | None
    max_dense_regions: int

    def __post_init__(self) -> None:
        for field in ("max_decoded_frames", "max_window_ms", "max_dense_regions"):
            if _non_negative(getattr(self, field), field) == 0:
                raise ContractError("decode budget limits must be positive")
        if self.max_decode_time_ms is not None and (
            isinstance(self.max_decode_time_ms, bool) or not isinstance(self.max_decode_time_ms, int) or self.max_decode_time_ms <= 0
        ):
            raise ContractError("max_decode_time_ms must be positive or null")


@dataclass(frozen=True)
class EvidenceContribution:
    source: str
    score: float
    detail: str = ""
    frame_id: int | None = None

    def __post_init__(self) -> None:
        _text(self.source, "evidence source")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not -1.0 <= self.score <= 1.0:
            raise ContractError("evidence score must be between -1 and 1")
        if self.frame_id is not None:
            _non_negative(self.frame_id, "evidence frame_id")


@dataclass(frozen=True)
class FrameSelection:
    video_id: str
    frame_id: int

    def __post_init__(self) -> None:
        _text(self.video_id, "video_id")
        _non_negative(self.frame_id, "frame_id")


def validate_trake_selection(selections: Sequence[FrameSelection], *, allow_order_exception: bool = False) -> tuple[FrameSelection, ...]:
    values = tuple(selections)
    if len(values) < 2:
        raise ContractError("TRAKE selection must contain at least two frames")
    if len({item.video_id for item in values}) != 1:
        raise ContractError("TRAKE selections must use the same video")
    if not allow_order_exception and any(a.frame_id >= b.frame_id for a, b in zip(values, values[1:])):
        raise ContractError("TRAKE frame_ids must be strictly increasing")
    return values


@dataclass(frozen=True)
class RefineRequest:
    candidate: CoarseCandidate
    video_path: Path
    task: Task
    refinement_text: str
    policy: RefinementPolicy
    context: RefinementContext
    decode_budget: DecodeBudget
    evidence: tuple[EvidenceContribution, ...] = ()
    event_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.video_path, Path) or not str(self.video_path):
            raise ContractError("video_path must be a path")
        _text(self.refinement_text, "refinement_text")
        if not isinstance(self.task, Task) or not isinstance(self.policy, RefinementPolicy):
            raise ContractError("task and policy must use WP09 enums")
        if not isinstance(self.context, RefinementContext) or not isinstance(self.decode_budget, DecodeBudget):
            raise ContractError("context and decode_budget are required")
        if self.event_index is not None:
            _non_negative(self.event_index, "event_index")


@dataclass(frozen=True)
class ExactNeighborRequest:
    """Bounded request to inspect original consecutive frames from one anchor."""

    candidate: CoarseCandidate
    video_path: Path
    context: RefinementContext
    offsets: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not isinstance(self.video_path, Path) or not str(self.video_path):
            raise ContractError("video_path must be a path")
        if not isinstance(self.context, RefinementContext):
            raise ContractError("context is required")
        if not self.offsets or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in self.offsets):
            raise ContractError("offsets must be non-empty integer values")


@dataclass(frozen=True)
class ExactFrameHypothesis:
    video_id: str
    frame_id: int
    timestamp_ms: int
    score: float | None
    reason: str
    visual_score: float | None = None
    policy_score: float | None = None
    window_start_ms: int | None = None
    window_end_ms: int | None = None
    evidence: tuple[EvidenceContribution, ...] = ()


@dataclass(frozen=True)
class RefinementAudit:
    context: RefinementContext
    before_frame_id: int
    before_score: float | None
    after_frame_id: int | None
    after_score: float | None
    window_start_ms: int | None
    window_end_ms: int | None
    model_version: str
    config_version: str
    latency_ms: float
    cache_hit: bool
    decoded_frame_count: int


@dataclass(frozen=True)
class RefineResult:
    coarse_candidate: CoarseCandidate
    hypotheses: tuple[ExactFrameHypothesis, ...]
    degraded_reason: str | None
    status: RefinementStatus
    context: RefinementContext
    audit: RefinementAudit
    modality_provenance: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        """JSON-safe handoff record; the caller owns selection/submission."""
        return asdict(self)
