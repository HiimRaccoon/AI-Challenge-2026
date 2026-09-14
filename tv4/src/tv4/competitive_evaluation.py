"""QA/TRAKE benchmark, ablation, and mandatory failure taxonomy.

Semantic QA correctness remains an externally adjudicated boolean in the
input.  This module intentionally does not approximate it with string match.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Mapping, Sequence


QA_FAILURES = {
    "QA-A": "video_not_retrieved",
    "QA-B": "correct_video_wrong_time",
    "QA-C": "correct_time_missing_evidence",
    "QA-D": "correct_evidence_wrong_answer",
    "QA-E": "correct_answer_wrong_frame",
    "QA-F": "operator_or_ui_problem",
}
TRAKE_FAILURES = {
    "TR-A": "event_parse_wrong",
    "TR-B": "video_shortlist_wrong",
    "TR-C": "event_candidate_recall_low",
    "TR-D": "sequence_alignment_wrong",
    "TR-E": "verifier_wrong",
    "TR-F": "final_frame_wrong",
}
_QA_STAGE_FIELDS = (
    "video_retrieved", "time_retrieved", "evidence_sufficient",
    "answer_correct", "frame_acceptable", "operator_ok",
)
_TRAKE_STAGE_FIELDS = (
    "event_parse_correct", "video_shortlisted", "event_candidates_recalled",
    "sequence_aligned", "verifier_correct", "final_frames_acceptable",
)


def classify_qa_failure(stages: Mapping[str, object]) -> str | None:
    checks = (
        ("video_retrieved", "QA-A"),
        ("time_retrieved", "QA-B"),
        ("evidence_sufficient", "QA-C"),
        ("answer_correct", "QA-D"),
        ("frame_acceptable", "QA-E"),
        ("operator_ok", "QA-F"),
    )
    return next((code for field, code in checks if stages.get(field) is False), None)


def classify_trake_failure(stages: Mapping[str, object]) -> str | None:
    checks = (
        ("event_parse_correct", "TR-A"),
        ("video_shortlisted", "TR-B"),
        ("event_candidates_recalled", "TR-C"),
        ("sequence_aligned", "TR-D"),
        ("verifier_correct", "TR-E"),
        ("final_frames_acceptable", "TR-F"),
    )
    return next((code for field, code in checks if stages.get(field) is False), None)


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _in_interval(value: object, interval: Mapping[str, object], prefix: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    start = interval.get(f"start_{prefix}")
    end = interval.get(f"end_{prefix}")
    return (
        isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and start <= value <= end
    )


def _frame_hit(prediction: Mapping[str, object], truth: Mapping[str, object]) -> bool:
    if prediction.get("video_id") != truth.get("video_id"):
        return False
    intervals = truth.get("acceptable_frame_intervals", ())
    return isinstance(intervals, Sequence) and any(
        isinstance(interval, Mapping) and _in_interval(prediction.get("frame_id"), interval, "frame_id")
        for interval in intervals
    )


def _time_hit(prediction: Mapping[str, object], truth: Mapping[str, object]) -> bool:
    if prediction.get("video_id") != truth.get("video_id"):
        return False
    interval = truth.get("answer_window_ms")
    return isinstance(interval, Mapping) and _in_interval(prediction.get("timestamp_ms"), interval, "ms")


def evaluate_qa_case(record: Mapping[str, object]) -> dict[str, float]:
    truth = record.get("ground_truth")
    raw_predictions = record.get("predictions", ())
    if not isinstance(truth, Mapping) or not isinstance(raw_predictions, Sequence):
        raise ValueError("QA record requires ground_truth and predictions")
    predictions = [item for item in raw_predictions if isinstance(item, Mapping)]
    top1 = predictions[:1]
    top5 = predictions[:5]
    top10 = predictions[:10]
    joint = lambda item: _frame_hit(item, truth) and item.get("answer_correct") is True
    verifier_positive = [item for item in top5 if item.get("independently_supported") is True]
    verifier_true = sum(1 for item in verifier_positive if joint(item))
    return {
        "primary_joint_top5": float(any(joint(item) for item in top5)),
        "correct_video_recall_at_5": float(any(item.get("video_id") == truth.get("video_id") for item in top5)),
        "answer_window_recall_at_10": float(any(_time_hit(item, truth) for item in top10)),
        "frame_hit_at_1": float(any(_frame_hit(item, truth) for item in top1)),
        "frame_hit_at_5": float(any(_frame_hit(item, truth) for item in top5)),
        "semantic_answer_accuracy": float(bool(top1 and top1[0].get("answer_correct") is True)),
        "verifier_true_positives": float(verifier_true),
        "verifier_positives": float(len(verifier_positive)),
        "operator_time_ms": _number(record.get("operator_time_ms")),
        "latency_ms": _number(record.get("latency_ms")),
    }


def _trake_event_hits(
    prediction: Mapping[str, object], truth: Mapping[str, object],
) -> list[bool]:
    intervals = truth.get("event_frame_intervals", ())
    frames = prediction.get("frame_ids", ())
    if prediction.get("video_id") != truth.get("video_id"):
        return [False for _ in intervals] if isinstance(intervals, Sequence) else []
    if not isinstance(intervals, Sequence) or not isinstance(frames, Sequence):
        return []
    return [
        isinstance(interval, Mapping) and _in_interval(frame_id, interval, "frame_id")
        for frame_id, interval in zip(frames, intervals)
    ]


def _event_recall(
    predictions: Sequence[Mapping[str, object]], truth: Mapping[str, object], k: int,
) -> float:
    intervals = truth.get("event_frame_intervals", ())
    if not isinstance(intervals, Sequence) or not intervals:
        return 0.0
    hits = [False] * len(intervals)
    for prediction in predictions[:k]:
        for index, matched in enumerate(_trake_event_hits(prediction, truth)):
            hits[index] = hits[index] or matched
    return sum(hits) / len(hits)


def _alternative_recall(prediction: Mapping[str, object], truth: Mapping[str, object]) -> float:
    alternatives = prediction.get("event_alternatives", ())
    intervals = truth.get("event_frame_intervals", ())
    if not isinstance(alternatives, Sequence) or not isinstance(intervals, Sequence) or not intervals:
        return 0.0
    hits = []
    for event_alternatives, interval in zip(alternatives, intervals):
        candidates = event_alternatives if isinstance(event_alternatives, Sequence) else ()
        hits.append(any(
            isinstance(item, Mapping)
            and item.get("video_id", prediction.get("video_id")) == truth.get("video_id")
            and isinstance(interval, Mapping)
            and _in_interval(item.get("frame_id"), interval, "frame_id")
            for item in candidates[:5]
        ))
    return sum(hits) / len(intervals)


def evaluate_trake_case(record: Mapping[str, object], *, near_threshold: float = 0.75) -> dict[str, float]:
    truth = record.get("ground_truth")
    raw_predictions = record.get("predictions", ())
    if not isinstance(truth, Mapping) or not isinstance(raw_predictions, Sequence):
        raise ValueError("TRAKE record requires ground_truth and predictions")
    predictions = [item for item in raw_predictions if isinstance(item, Mapping)]
    event_count = len(truth.get("event_frame_intervals", ()))
    hit_rates = [
        sum(_trake_event_hits(item, truth)) / event_count if event_count else 0.0
        for item in predictions[:5]
    ]
    exact = [rate == 1.0 for rate in hit_rates]
    top1 = predictions[0] if predictions else {}
    timestamps = top1.get("timestamps_ms", ())
    strict_order = (
        isinstance(timestamps, Sequence)
        and len(timestamps) == event_count
        and all(
            isinstance(left, int) and not isinstance(left, bool)
            and isinstance(right, int) and not isinstance(right, bool)
            and left < right
            for left, right in zip(timestamps, timestamps[1:])
        )
    )
    event_video_ids = top1.get("event_video_ids", ())
    same_video = (
        all(value == top1.get("video_id") for value in event_video_ids)
        if isinstance(event_video_ids, Sequence) and event_video_ids
        else bool(top1)
    )
    return {
        "primary_near_sequence_top5": float(any(rate >= near_threshold for rate in hit_rates)),
        "correct_video_recall_at_5": float(any(item.get("video_id") == truth.get("video_id") for item in predictions[:5])),
        "per_event_recall_at_5": _event_recall(predictions, truth, 5),
        "per_event_recall_at_10": _event_recall(predictions, truth, 10),
        "event_alternative_recall_at_5": _alternative_recall(top1, truth) if top1 else 0.0,
        "full_sequence_hit_at_1": float(bool(exact[:1] and exact[0])),
        "full_sequence_hit_at_5": float(any(exact)),
        "mean_event_hit_rate": hit_rates[0] if hit_rates else 0.0,
        "same_video_validity": float(same_video),
        "strict_order_validity": float(strict_order),
        "operator_time_ms": _number(record.get("operator_time_ms")),
        "latency_ms": _number(record.get("latency_ms")),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_dataset(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    task_rows: dict[str, list[dict[str, float]]] = {"QA": [], "TRAKE": []}
    failures: Counter[str] = Counter()
    failure_records: list[dict[str, str]] = []
    for record in records:
        task = str(record.get("task", "")).upper()
        stages = record.get("stages")
        if not isinstance(stages, Mapping):
            raise ValueError(f"{task or 'record'} requires mandatory stage booleans")
        if task == "QA":
            if any(not isinstance(stages.get(field), bool) for field in _QA_STAGE_FIELDS):
                raise ValueError("QA record requires all mandatory QA stage booleans")
            task_rows[task].append(evaluate_qa_case(record))
            failure = classify_qa_failure(stages)
        elif task == "TRAKE":
            if any(not isinstance(stages.get(field), bool) for field in _TRAKE_STAGE_FIELDS):
                raise ValueError("TRAKE record requires all mandatory TRAKE stage booleans")
            task_rows[task].append(evaluate_trake_case(record))
            failure = classify_trake_failure(stages)
        else:
            raise ValueError(f"unsupported task: {task}")
        if failure:
            failures[failure] += 1
            failure_records.append({"query_id": str(record.get("query_id", "")), "task": task, "failure": failure})

    report: dict[str, object] = {"tasks": {}, "failure_counts": dict(sorted(failures.items())), "failures": failure_records}
    primary_values: list[float] = []
    latencies: list[float] = []
    for task, rows in task_rows.items():
        if not rows:
            continue
        metric_names = rows[0].keys()
        aggregate = {name: mean(row[name] for row in rows) for name in metric_names}
        verifier_positives = sum(row.get("verifier_positives", 0.0) for row in rows)
        if task == "QA":
            aggregate["verifier_precision"] = (
                sum(row["verifier_true_positives"] for row in rows) / verifier_positives
                if verifier_positives else 0.0
            )
            aggregate.pop("verifier_true_positives", None)
            aggregate.pop("verifier_positives", None)
            primary_values.extend(row["primary_joint_top5"] for row in rows)
        else:
            primary_values.extend(row["primary_near_sequence_top5"] for row in rows)
        task_latencies = [row["latency_ms"] for row in rows]
        aggregate["latency_p50_ms"] = _percentile(task_latencies, 0.50)
        aggregate["latency_p95_ms"] = _percentile(task_latencies, 0.95)
        latencies.extend(task_latencies)
        report["tasks"][task] = {"count": len(rows), "metrics": {key: round(value, 6) for key, value in aggregate.items()}}
    report["overall_primary"] = round(mean(primary_values), 6) if primary_values else 0.0
    report["overall_latency_p95_ms"] = round(_percentile(latencies, 0.95), 6)
    return report


def build_ablation_report(
    variants: Mapping[str, Sequence[Mapping[str, object]]], baseline_id: str,
) -> dict[str, object]:
    if baseline_id not in variants:
        raise ValueError("baseline_id is absent from variants")
    reports = {name: evaluate_dataset(records) for name, records in variants.items()}
    baseline = reports[baseline_id]
    baseline_quality = float(baseline["overall_primary"])
    baseline_latency = float(baseline["overall_latency_p95_ms"])
    rows = []
    for name, report in reports.items():
        quality = float(report["overall_primary"])
        latency = float(report["overall_latency_p95_ms"])
        rows.append({
            "variant": name,
            "is_baseline": name == baseline_id,
            "overall_primary": quality,
            "quality_delta": round(quality - baseline_quality, 6),
            "latency_p95_ms": latency,
            "latency_delta_ms": round(latency - baseline_latency, 6),
            "report": report,
        })
    return {"baseline_id": baseline_id, "variants": rows}
