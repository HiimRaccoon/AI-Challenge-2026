"""Framework-free contracts for feedback session input and output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


class FeedbackValidationError(ValueError):
    """A feedback request violates the deterministic WP08 contract."""


class RevisionConflict(RuntimeError):
    """A mutation was computed from an obsolete ranking revision."""


class SessionExpired(RuntimeError):
    """A session passed its fixed retention deadline."""


class ModelRankingFailed(RuntimeError):
    """One of the required model paths could not produce a complete ranking."""


@dataclass(frozen=True)
class CandidateId:
    video_id: str
    frame_id: int

    def __post_init__(self) -> None:
        if not self.video_id.strip() or self.frame_id < 0:
            raise FeedbackValidationError("candidate identity is invalid")


@dataclass(frozen=True)
class FeedbackEvent:
    candidate_id: CandidateId
    feedback_text: str

    @classmethod
    def create(cls, *, candidate_id: CandidateId, feedback_text: str) -> "FeedbackEvent":
        if not isinstance(feedback_text, str) or not feedback_text.strip():
            raise FeedbackValidationError("feedback text must contain a non-whitespace character")
        if len(feedback_text) > 300:
            raise FeedbackValidationError("feedback text must not exceed 300 raw characters")
        return cls(candidate_id=candidate_id, feedback_text=feedback_text)


@dataclass(frozen=True)
class Confirmation:
    confirmation_id: str
    session_id: str
    revision: int
    candidate_id: CandidateId
    created_at_utc: str


@dataclass(frozen=True)
class RenderedCandidate:
    candidate_id: CandidateId
    display_rank: int
    timestamp_ms: int | None = None
    keyframe_path: str | None = None


@dataclass(frozen=True)
class CandidateMetadata:
    """Media data carried to TV5 separately from a ranking identity."""

    candidate_id: CandidateId
    timestamp_ms: int
    keyframe_path: str

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or not self.keyframe_path:
            raise FeedbackValidationError("candidate media metadata is invalid")


@dataclass(frozen=True)
class SessionPool:
    """A validated, serializable WP03 pool supplied when a session starts.

    ``snapshot`` is deliberately opaque to the service: it is persisted verbatim
    and handed back to the WP03 ranking/rendering adapter after a process restart.
    """

    wp03_run_id: str
    candidates: tuple[CandidateId, ...]
    candidate_metadata: tuple[CandidateMetadata, ...]
    snapshot: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.wp03_run_id or not self.candidates:
            raise FeedbackValidationError("session pool is invalid")
        if len(set(self.candidates)) != len(self.candidates):
            raise FeedbackValidationError("session pool contains duplicate candidates")
        metadata = {item.candidate_id for item in self.candidate_metadata}
        if metadata != set(self.candidates):
            raise FeedbackValidationError("session pool metadata does not match candidates")


@dataclass(frozen=True)
class StableFeedbackConfig:
    """Approved configuration required to expose a composed runtime as stable."""

    model_keys: tuple[str, str, str, str]
    benchmark_run_id: str
    approved_at_utc: str
    fusion_alpha: float = 0.75
    rrf_k: int = 60
    diversity_dedup_window_ms: int = 1_000
    text_template_version: str = "wp08-v1"

    def __post_init__(self) -> None:
        if len(self.model_keys) != 4 or len(set(self.model_keys)) != 4 or any(not key.strip() for key in self.model_keys):
            raise FeedbackValidationError("stable feedback requires exactly four unique model keys")
        if not self.benchmark_run_id.strip() or not self.approved_at_utc.strip():
            raise FeedbackValidationError("stable feedback requires approved benchmark provenance")
        try:
            datetime.fromisoformat(self.approved_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise FeedbackValidationError("benchmark approval timestamp is invalid") from exc
        if not 0.0 <= self.fusion_alpha <= 1.0 or self.rrf_k < 1 or self.diversity_dedup_window_ms < 0:
            raise FeedbackValidationError("stable feedback configuration is invalid")
        if not self.text_template_version.strip():
            raise FeedbackValidationError("text template version is required")


@dataclass(frozen=True)
class FirstCorrect:
    session_id: str
    revision: int
    candidate_id: CandidateId
    cohort: str
    recorded_at_utc: str
    elapsed_ms: int

    def __post_init__(self) -> None:
        if self.cohort not in {"no_feedback", "with_feedback"} or self.elapsed_ms < 0:
            raise FeedbackValidationError("first-correct record is invalid")


@dataclass(frozen=True)
class FeedbackMetricSummary:
    cohort: str
    count: int
    elapsed_ms_samples: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.cohort not in {"no_feedback", "with_feedback"} or self.count < 0 or self.count != len(self.elapsed_ms_samples):
            raise FeedbackValidationError("feedback metric summary is invalid")


@dataclass(frozen=True)
class SessionView:
    session_id: str
    revision: int
    candidates: tuple[RenderedCandidate, ...]
    expires_at_utc: str | None = None
    wp03_run_id: str | None = None
