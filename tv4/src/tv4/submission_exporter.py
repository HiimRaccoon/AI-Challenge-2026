"""Production handoff exporter for the retrieval team.

This module owns the boundary between TV4's rich internal results and the
minimal JSON contract consumed by the submission team.  It deliberately
keeps scores, evidence, ranks, and all other debug fields out of
``predictions_final.json``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .clients.tv1_client import TV1Client
from .contracts import SearchCandidate, TrakeHypothesis
from .wp11_vqa import VqaResult


_VIDEO_ID = re.compile(r"^L\d+_V\d+$")
_TASKS = {"KIS", "VQA", "TRAKE"}


class PredictionValidationError(ValueError):
    """The final handoff artifact is unsafe or violates its contract."""


@dataclass(frozen=True)
class QuerySpec:
    """The subset of a query definition required for final-output validation."""

    query_id: str
    task: str
    event_count: int | None = None


@dataclass(frozen=True)
class ValidationSummary:
    query_count: int
    kis_count: int
    vqa_count: int
    trake_count: int


class AuthoritativeFrameResolver:
    """Accept only frame IDs explicitly present in TV1's canonical mapping.

    Upstream retrieval already uses TV1 frame IDs.  The exporter must still
    perform this final membership check, rather than silently trusting a row
    index, PTS, or a model-local identifier.
    """

    def __init__(self, tv1: TV1Client) -> None:
        self._tv1 = tv1
        self._frame_ids_by_video: dict[str, set[int]] = {}

    def resolve(self, video_id: str, frame_id: int) -> int:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise PredictionValidationError(
                f"{video_id}: frame_id must be a non-negative integer, got {frame_id!r}"
            )
        frame_ids = self._frame_ids_by_video.get(video_id)
        if frame_ids is None:
            try:
                frames = self._tv1.frames(video_id)
                frame_ids = {int(frame["frame_id"]) for frame in frames}
            except Exception as exc:  # a resolver outage must fail the handoff, never be guessed
                raise PredictionValidationError(
                    f"{video_id}: could not load the authoritative TV1 frame mapping: {exc}"
                ) from exc
            self._frame_ids_by_video[video_id] = frame_ids
        if frame_id not in frame_ids:
            raise PredictionValidationError(
                f"{video_id}: frame_id {frame_id} is absent from the authoritative TV1 mapping"
            )
        return frame_id

    def resolve_candidate(self, candidate: SearchCandidate) -> int:
        return self.resolve(candidate.video_id, candidate.frame_id)


def query_specs_from_records(records: Sequence[Mapping[str, Any]]) -> list[QuerySpec]:
    """Create validation specs from TV4's current batch-query JSON format.

    The future owner-supplied manifest loader can construct the same
    ``QuerySpec`` objects, so the final-output contract remains unchanged.
    """
    specs: list[QuerySpec] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        query_id = record.get("query_id")
        task = record.get("task")
        if not isinstance(query_id, str) or not query_id.strip():
            raise PredictionValidationError(f"query at index {index} has no non-empty query_id")
        if query_id in seen:
            raise PredictionValidationError(f"duplicate query_id: {query_id}")
        if not isinstance(task, str) or task not in _TASKS:
            raise PredictionValidationError(f"{query_id}: task must be one of {sorted(_TASKS)}")
        event_count: int | None = None
        if task == "TRAKE":
            events = record.get("events")
            if not isinstance(events, list) or not events:
                raise PredictionValidationError(f"{query_id}: TRAKE requires a non-empty events list")
            event_count = len(events)
        specs.append(QuerySpec(query_id=query_id, task=task, event_count=event_count))
        seen.add(query_id)
    if not specs:
        raise PredictionValidationError("the query list is empty")
    return specs


def _check_video_and_frame(video: object, frame: object, *, context: str) -> None:
    if not isinstance(video, str) or not _VIDEO_ID.fullmatch(video):
        raise PredictionValidationError(f"{context}: video must match L<digits>_V<digits>, got {video!r}")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise PredictionValidationError(f"{context}: frame must be a non-negative integer, got {frame!r}")


def validate_predictions(
    predictions: Mapping[str, Any], query_specs: Sequence[QuerySpec]
) -> ValidationSummary:
    """Validate the minimal final JSON contract and raise on the first error."""
    if not isinstance(predictions, Mapping):
        raise PredictionValidationError("predictions must be a JSON object keyed by query ID")
    for spec in query_specs:
        if spec.task not in _TASKS:
            raise PredictionValidationError(f"{spec.query_id}: invalid task {spec.task!r}")
        if spec.task == "TRAKE" and (
            isinstance(spec.event_count, bool)
            or not isinstance(spec.event_count, int)
            or spec.event_count < 1
        ):
            raise PredictionValidationError(f"{spec.query_id}: TRAKE event count must be a positive integer")
    expected = {spec.query_id: spec for spec in query_specs}
    if len(expected) != len(query_specs):
        raise PredictionValidationError("query specs contain duplicate query IDs")
    actual = set(predictions)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise PredictionValidationError(
            f"prediction query set differs from expected; missing={missing}, unexpected={unexpected}"
        )

    counts = {"KIS": 0, "VQA": 0, "TRAKE": 0}
    for query_id, spec in expected.items():
        candidates = predictions[query_id]
        if not isinstance(candidates, list) or not (1 <= len(candidates) <= 100):
            raise PredictionValidationError(f"{query_id}: candidate count must be in [1, 100]")
        counts[spec.task] += 1
        for rank, candidate in enumerate(candidates, start=1):
            context = f"{query_id} candidate {rank}"
            if not isinstance(candidate, dict):
                raise PredictionValidationError(f"{context}: candidate must be an object")
            if spec.task == "KIS":
                if set(candidate) != {"video", "frame"}:
                    raise PredictionValidationError(f"{context}: KIS fields must be exactly video, frame")
                _check_video_and_frame(candidate["video"], candidate["frame"], context=context)
            elif spec.task == "VQA":
                if set(candidate) != {"video", "frame", "answer"}:
                    raise PredictionValidationError(f"{context}: VQA fields must be exactly video, frame, answer")
                _check_video_and_frame(candidate["video"], candidate["frame"], context=context)
                answer = candidate["answer"]
                if not isinstance(answer, str) or not answer:
                    raise PredictionValidationError(f"{context}: answer must be a non-empty string")
                if len(answer) > 100 or answer != answer.strip() or "\n" in answer or "\r" in answer:
                    raise PredictionValidationError(
                        f"{context}: answer must be <=100 chars with no surrounding whitespace or newline"
                    )
            else:
                if set(candidate) != {"video", "frames"}:
                    raise PredictionValidationError(f"{context}: TRAKE fields must be exactly video, frames")
                video = candidate["video"]
                frames = candidate["frames"]
                if not isinstance(video, str) or not _VIDEO_ID.fullmatch(video):
                    raise PredictionValidationError(f"{context}: invalid video {video!r}")
                if not isinstance(frames, list) or len(frames) != spec.event_count:
                    raise PredictionValidationError(
                        f"{context}: TRAKE requires exactly {spec.event_count} frames"
                    )
                for frame in frames:
                    _check_video_and_frame(video, frame, context=context)
                if any(left >= right for left, right in zip(frames, frames[1:])):
                    raise PredictionValidationError(f"{context}: TRAKE frames must be strictly increasing")

    return ValidationSummary(
        query_count=len(query_specs),
        kis_count=counts["KIS"],
        vqa_count=counts["VQA"],
        trake_count=counts["TRAKE"],
    )


def validate_resolved_predictions(
    predictions: Mapping[str, Any],
    query_specs: Sequence[QuerySpec],
    resolver: AuthoritativeFrameResolver,
) -> ValidationSummary:
    """Validate schema *and* prove that every final frame is from TV1.

    This is deliberately callable without running retrieval, so the exact
    ``predictions_final.json`` handed to another owner can be independently
    re-checked before it is accepted.
    """
    summary = validate_predictions(predictions, query_specs)
    specs_by_id = {spec.query_id: spec for spec in query_specs}
    for query_id, candidates in predictions.items():
        spec = specs_by_id[query_id]
        for candidate in candidates:
            if spec.task in {"KIS", "VQA"}:
                resolver.resolve(candidate["video"], candidate["frame"])
            else:
                for frame_id in candidate["frames"]:
                    resolver.resolve(candidate["video"], frame_id)
    return summary


def build_predictions_final(
    query_specs: Sequence[QuerySpec],
    results_by_query: Mapping[str, object],
    resolver: AuthoritativeFrameResolver,
) -> dict[str, list[dict[str, Any]]]:
    """Convert ranked TV4 results to final JSON without changing their order."""
    expected = {spec.query_id for spec in query_specs}
    actual = set(results_by_query)
    if actual != expected:
        raise PredictionValidationError(
            f"pipeline result set differs from expected; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    predictions: dict[str, list[dict[str, Any]]] = {}
    for spec in query_specs:
        result = results_by_query[spec.query_id]
        if spec.task == "KIS":
            if not isinstance(result, list):
                raise PredictionValidationError(f"{spec.query_id}: KIS result is not a ranked list")
            rows: list[dict[str, Any]] = []
            for candidate in result[:100]:
                if not isinstance(candidate, SearchCandidate):
                    raise PredictionValidationError(f"{spec.query_id}: KIS result has an invalid row")
                rows.append({"video": candidate.video_id, "frame": resolver.resolve_candidate(candidate)})
            predictions[spec.query_id] = rows
        elif spec.task == "VQA":
            if not isinstance(result, list):
                raise PredictionValidationError(f"{spec.query_id}: VQA result is not a ranked list")
            rows: list[dict[str, Any]] = []
            for item in result[:100]:
                if not isinstance(item, VqaResult):
                    raise PredictionValidationError(f"{spec.query_id}: VQA result has an invalid row")
                if item.manual_fallback or not item.verified:
                    raise PredictionValidationError(
                        f"{spec.query_id}: VQA candidate requires manual review and cannot enter final output"
                    )
                rows.append(
                    {
                        "video": item.candidate.video_id,
                        "frame": resolver.resolve_candidate(item.candidate),
                        "answer": item.answer,
                    }
                )
            predictions[spec.query_id] = rows
        else:
            if not isinstance(result, TrakeHypothesis):
                raise PredictionValidationError(f"{spec.query_id}: no valid TRAKE alignment was produced")
            if spec.event_count is None:
                raise PredictionValidationError(f"{spec.query_id}: TRAKE event count is missing")
            frames = [resolver.resolve(result.video_id, frame_id) for frame_id in result.frame_ids]
            predictions[spec.query_id] = [{"video": result.video_id, "frames": frames}]

    validate_resolved_predictions(predictions, query_specs, resolver)
    return predictions


def write_predictions_final(
    path: Path,
    query_specs: Sequence[QuerySpec],
    results_by_query: Mapping[str, object],
    resolver: AuthoritativeFrameResolver,
) -> ValidationSummary:
    """Build, validate, and atomically publish ``predictions_final.json``."""
    predictions = build_predictions_final(query_specs, results_by_query, resolver)
    summary = validate_resolved_predictions(predictions, query_specs, resolver)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return summary
