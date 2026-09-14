"""Bridge from the Retrieval Owner's predictions_final.json contract
(README_RETRIEVAL_HANDOFF_AIC26_v2) into this package's own submission
model (KisPrediction / VqaPrediction / TrakePrediction -> CSV -> ZIP).

predictions_final.json has already passed submission_exporter's fail-fast
validation before it reaches here (missing/extra queries, candidate counts,
frame identity, VQA `verified` gate, TRAKE chronology, ...). That prior gate
is treated as the operator-approval-equivalent for VQA answers: an answer
only appears in predictions_final.json if it was machine-verified against
evidence, which is the same bar this package's own `approved` flag exists to
enforce for UI-approved answers. No answer text or frame identity is
recomputed here -- this module only reshapes already-accepted data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import KisPrediction, TrakePrediction, ValidationReport, VqaPrediction
from .csv_exporter import export_kis_csv, export_trake_csv, export_vqa_csv
from .packager import package_submission_zip


class PredictionsFinalError(ValueError):
    """predictions_final.json is missing, malformed, or fails this bridge's own checks."""


def _infer_task(query_id: str, candidates: list[dict[str, Any]]) -> str:
    """Infer KIS/VQA/TRAKE from candidate shape, matching submission_exporter's schema."""
    if not candidates:
        raise PredictionsFinalError(f"{query_id}: has zero candidates")
    keys = set(candidates[0])
    if keys == {"video", "frame"}:
        return "KIS"
    if keys == {"video", "frame", "answer"}:
        return "VQA"
    if keys == {"video", "frames"}:
        return "TRAKE"
    raise PredictionsFinalError(f"{query_id}: candidate fields {sorted(keys)} match no known task shape")


def predictions_to_query_csvs(predictions: Mapping[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Convert a loaded predictions_final.json into {query_id: csv_text}."""
    if not isinstance(predictions, Mapping):
        raise PredictionsFinalError("predictions_final.json root must be a JSON object")

    csv_by_query: dict[str, str] = {}
    for query_id, candidates in predictions.items():
        if not isinstance(candidates, list):
            raise PredictionsFinalError(f"{query_id}: candidates must be a list")
        task = _infer_task(query_id, candidates)

        if task == "KIS":
            items = [
                KisPrediction(video_id=c["video"], frame_id=c["frame"], rank=rank)
                for rank, c in enumerate(candidates, start=1)
            ]
            csv_by_query[query_id] = export_kis_csv(items)
        elif task == "VQA":
            items = [
                VqaPrediction(
                    video_id=c["video"], frame_id=c["frame"], approved_answer=c["answer"],
                    rank=rank, approved=True,
                )
                for rank, c in enumerate(candidates, start=1)
            ]
            csv_by_query[query_id] = export_vqa_csv(items)
        else:  # TRAKE
            items = [
                TrakePrediction(
                    video_id=c["video"], event_frame_ids=tuple(c["frames"]),
                    rank=rank, expected_event_count=len(c["frames"]),
                )
                for rank, c in enumerate(candidates, start=1)
            ]
            csv_by_query[query_id] = export_trake_csv(items)

    if not csv_by_query:
        raise PredictionsFinalError("predictions_final.json contains no queries")
    return csv_by_query


def build_submission_zip_from_predictions_final(
    predictions_final_path: Path, output_zip_path: Path,
) -> ValidationReport:
    """Read predictions_final.json and write a validated submission/ ZIP."""
    try:
        predictions = json.loads(predictions_final_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PredictionsFinalError(f"cannot read {predictions_final_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PredictionsFinalError(f"{predictions_final_path} is not valid JSON: {exc}") from exc

    csv_by_query = predictions_to_query_csvs(predictions)
    return package_submission_zip(csv_by_query, output_zip_path)
