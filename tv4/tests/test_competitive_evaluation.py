from __future__ import annotations

import pytest

from tv4.competitive_evaluation import (
    build_ablation_report,
    classify_qa_failure,
    classify_trake_failure,
    evaluate_dataset,
)


def _qa_record(*, hit: bool, latency_ms: int = 100):
    return {
        "query_id": "qa-1",
        "task": "QA",
        "latency_ms": latency_ms,
        "operator_time_ms": 1_000,
        "ground_truth": {
            "video_id": "V1",
            "answer_window_ms": {"start_ms": 900, "end_ms": 1_100},
            "acceptable_frame_intervals": [{"start_frame_id": 9, "end_frame_id": 11}],
        },
        "predictions": [{
            "video_id": "V1" if hit else "V2",
            "frame_id": 10,
            "timestamp_ms": 1_000,
            "answer_correct": hit,
            "independently_supported": True,
        }],
        "stages": {
            "video_retrieved": hit,
            "time_retrieved": hit,
            "evidence_sufficient": hit,
            "answer_correct": hit,
            "frame_acceptable": hit,
            "operator_ok": True,
        },
    }


def _trake_record(*, hit: bool, latency_ms: int = 200):
    return {
        "query_id": "tr-1",
        "task": "TRAKE",
        "latency_ms": latency_ms,
        "ground_truth": {
            "video_id": "V1",
            "event_frame_intervals": [
                {"start_frame_id": 9, "end_frame_id": 11},
                {"start_frame_id": 19, "end_frame_id": 21},
            ],
        },
        "predictions": [{
            "video_id": "V1" if hit else "V2",
            "frame_ids": [10, 20],
            "timestamps_ms": [1_000, 2_000],
            "event_alternatives": [
                [{"video_id": "V1", "frame_id": 10}],
                [{"video_id": "V1", "frame_id": 20}],
            ],
        }],
        "stages": {
            "event_parse_correct": True,
            "video_shortlisted": hit,
            "event_candidates_recalled": hit,
            "sequence_aligned": hit,
            "verifier_correct": hit,
            "final_frames_acceptable": hit,
        },
    }


def test_failure_taxonomy_returns_first_failed_stage():
    assert classify_qa_failure({
        "video_retrieved": True,
        "time_retrieved": True,
        "evidence_sufficient": False,
        "answer_correct": False,
    }) == "QA-C"
    assert classify_trake_failure({
        "event_parse_correct": True,
        "video_shortlisted": True,
        "event_candidates_recalled": True,
        "sequence_aligned": False,
        "verifier_correct": False,
    }) == "TR-D"


def test_dataset_report_calculates_readme_qa_and_trake_metrics():
    report = evaluate_dataset((_qa_record(hit=True), _trake_record(hit=True)))

    assert report["overall_primary"] == 1.0
    assert report["tasks"]["QA"]["metrics"]["primary_joint_top5"] == 1.0
    assert report["tasks"]["QA"]["metrics"]["verifier_precision"] == 1.0
    assert report["tasks"]["TRAKE"]["metrics"]["full_sequence_hit_at_1"] == 1.0
    assert report["tasks"]["TRAKE"]["metrics"]["strict_order_validity"] == 1.0
    assert report["failure_counts"] == {}


def test_ablation_report_exposes_quality_and_latency_deltas():
    variants = {
        "baseline": [_qa_record(hit=False, latency_ms=100), _trake_record(hit=False, latency_ms=200)],
        "competitive": [_qa_record(hit=True, latency_ms=130), _trake_record(hit=True, latency_ms=240)],
    }

    report = build_ablation_report(variants, "baseline")
    rows = {row["variant"]: row for row in report["variants"]}

    assert rows["baseline"]["quality_delta"] == 0.0
    assert rows["competitive"]["quality_delta"] == 1.0
    assert rows["competitive"]["latency_delta_ms"] > 0
    assert rows["competitive"]["report"]["failure_counts"] == {}
    assert rows["baseline"]["report"]["failure_counts"] == {"QA-A": 1, "TR-B": 1}


def test_dataset_rejects_missing_mandatory_failure_stage_flags():
    record = _qa_record(hit=False)
    record["stages"] = {}

    with pytest.raises(ValueError, match="mandatory QA stage"):
        evaluate_dataset((record,))
