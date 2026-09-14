"""WP13 evaluation, metric calculation, and report ingestion package."""
from __future__ import annotations

from .metrics import (
    evaluate_kis_prediction,
    evaluate_vqa_prediction,
    evaluate_trake_prediction,
    calculate_r_at_k,
    calculate_final_score,
    EvaluationResult,
    GroundTruthInterval,
)
from .reports import (
    BenchmarkReport,
    AblationReport,
    ErrorAnalysisReport,
    MockCompetitionReport,
)
from .preprocessing_reports import (
    PreprocessingReport,
    ingest_preprocessing_run_report,
)

__all__ = [
    "evaluate_kis_prediction",
    "evaluate_vqa_prediction",
    "evaluate_trake_prediction",
    "calculate_r_at_k",
    "calculate_final_score",
    "EvaluationResult",
    "GroundTruthInterval",
    "BenchmarkReport",
    "AblationReport",
    "ErrorAnalysisReport",
    "MockCompetitionReport",
    "PreprocessingReport",
    "ingest_preprocessing_run_report",
]
