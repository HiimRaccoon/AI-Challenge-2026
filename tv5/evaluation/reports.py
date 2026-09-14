"""Evaluation report schemas for Benchmark, Ablation, Error Analysis, and Mock Competitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .metrics import calculate_r_at_k, calculate_final_score


@dataclass(frozen=True)
class QueryEvaluationRecord:
    query_id: str
    task_type: str
    r_at_k: dict[int, float]
    final_score: float
    status: str = "COMPLETED"
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkReport:
    report_id: str
    config_id: str
    run_id: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mean_final_score: float = 0.0
    mean_r_at_k: dict[int, float] = field(default_factory=dict)
    query_records: tuple[QueryEvaluationRecord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AblationRecord:
    ablation_name: str
    disabled_modality: str
    mean_final_score: float
    score_delta: float


@dataclass(frozen=True)
class AblationReport:
    report_id: str
    baseline_config_id: str
    baseline_final_score: float
    ablations: tuple[AblationRecord, ...]
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorAnalysisRecord:
    query_id: str
    task_type: str
    failure_category: str  # e.g., "WRONG_VIDEO", "OUT_OF_INTERVAL", "UNAPPROVED_VQA", "TRAKE_REORDER"
    details: str
    suggested_action: str


@dataclass(frozen=True)
class ErrorAnalysisReport:
    report_id: str
    total_queries: int
    failed_queries_count: int
    records: tuple[ErrorAnalysisRecord, ...]
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MockCompetitionReport:
    mock_run_id: str
    iteration: int
    p0_issues_count: int
    overall_final_score: float
    kis_score: float
    vqa_score: float
    trake_score: float
    sign_off_status: str  # "PASS" | "FAIL_RETRY"
    operator_notes: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
