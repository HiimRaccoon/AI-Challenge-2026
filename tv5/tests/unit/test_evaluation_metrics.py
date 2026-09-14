"""Tests for official competition metrics, report schemas, and read-only report ingestion."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tv5.submission.contracts import KisPrediction, VqaPrediction, TrakePrediction
from tv5.evaluation import (
    evaluate_kis_prediction,
    evaluate_vqa_prediction,
    evaluate_trake_prediction,
    calculate_r_at_k,
    calculate_final_score,
    GroundTruthInterval,
    BenchmarkReport,
    AblationReport,
    ErrorAnalysisReport,
    MockCompetitionReport,
    ingest_preprocessing_run_report,
)


def test_kis_evaluation_interval_match_and_mismatch() -> None:
    gt = [GroundTruthInterval(video_id="L01_V001", start_frame_id=100, end_frame_id=200)]

    # Match inside interval
    pred_hit = KisPrediction(video_id="L01_V001", frame_id=150)
    res_hit = evaluate_kis_prediction(pred_hit, gt)
    assert res_hit.r_score == 1.0
    assert res_hit.is_hit is True

    # Out of interval
    pred_miss = KisPrediction(video_id="L01_V001", frame_id=250)
    assert evaluate_kis_prediction(pred_miss, gt).r_score == 0.0

    # Wrong video
    pred_wrong_vid = KisPrediction(video_id="L01_V002", frame_id=150)
    assert evaluate_kis_prediction(pred_wrong_vid, gt).r_score == 0.0


def test_vqa_evaluation_requires_semantic_adjudicator() -> None:
    gt = [GroundTruthInterval(video_id="L01_V001", start_frame_id=100, end_frame_id=200)]
    pred = VqaPrediction(video_id="L01_V001", frame_id=150, approved_answer="cốc màu đỏ", approved=True)

    # Without adjudicator -> INCOMPLETE
    res_no_adj = evaluate_vqa_prediction(pred, gt, semantic_adjudicator=None)
    assert res_no_adj.status == "INCOMPLETE / EXTERNAL ADJUDICATION REQUIRED"

    # With positive adjudicator -> 1.0
    res_pos = evaluate_vqa_prediction(pred, gt, semantic_adjudicator=lambda ans: "đỏ" in ans)
    assert res_pos.r_score == 1.0
    assert res_pos.is_hit is True

    # With negative adjudicator -> 0.0
    res_neg = evaluate_vqa_prediction(pred, gt, semantic_adjudicator=lambda ans: "xanh" in ans)
    assert res_neg.r_score == 0.0


def test_trake_evaluation_r_score_and_wrong_video() -> None:
    gt_events = [
        GroundTruthInterval("L01_V001", 100, 150),
        GroundTruthInterval("L01_V001", 200, 250),
        GroundTruthInterval("L01_V001", 300, 350),
        GroundTruthInterval("L01_V001", 400, 450),
    ]

    # 3 out of 4 matched => 0.75
    pred_3_of_4 = TrakePrediction("L01_V001", event_frame_ids=(120, 220, 320, 999), expected_event_count=4)
    res = evaluate_trake_prediction(pred_3_of_4, gt_events)
    assert res.r_score == 0.75

    # Wrong video => 0.0
    pred_wrong = TrakePrediction("L01_V002", event_frame_ids=(120, 220, 320, 420), expected_event_count=4)
    assert evaluate_trake_prediction(pred_wrong, gt_events).r_score == 0.0


def test_organizer_golden_final_score_0_74() -> None:
    """Golden case: R@1=0.5, R@5=0.8, R@20=0.8, R@50=0.8, R@100=0.8 -> Final = 0.74."""
    # Construct sequence where item 1 has score 0.5, item 2 has score 0.8, rest lower
    ranked_scores = [0.5, 0.8] + [0.1] * 98
    r_at_k = calculate_r_at_k(ranked_scores)
    assert r_at_k[1] == 0.5
    assert r_at_k[5] == 0.8
    assert r_at_k[20] == 0.8
    assert r_at_k[50] == 0.8
    assert r_at_k[100] == 0.8

    final_score = calculate_final_score(r_at_k)
    assert final_score == 0.74


def test_read_only_preprocessing_report_ingestion(tmp_path: Path) -> None:
    # 1. Non-existent dir
    rep_missing = ingest_preprocessing_run_report(tmp_path / "missing_dir")
    assert rep_missing.status == "ACTUALLY MISSING"

    # 2. Corrupt manifest
    corrupt_dir = tmp_path / "corrupt_run"
    corrupt_dir.mkdir()
    (corrupt_dir / "manifest.json").write_text("invalid json")
    rep_corrupt = ingest_preprocessing_run_report(corrupt_dir)
    assert rep_corrupt.status == "INCOMPATIBLE"

    # 3. Valid run
    valid_dir = tmp_path / "valid_run"
    (valid_dir / "manifest").mkdir(parents=True)
    manifest_data = {
        "preprocess_run_id": "run_v1_batch1",
        "videos": ["L01_V001", "L01_V002"],
        "keyframe_count": 500,
        "thumbnail_count": 500,
        "storage_bytes": 1048576,
        "throughput_fps": 120.5,
    }
    (valid_dir / "manifest" / "corpus_manifest.json").write_text(json.dumps(manifest_data))

    rep_valid = ingest_preprocessing_run_report(valid_dir, expected_run_id="run_v1_batch1")
    assert rep_valid.is_valid is True
    assert rep_valid.status == "READY"
    assert rep_valid.video_count == 2
    assert rep_valid.keyframe_count == 500
    assert rep_valid.throughput_fps == 120.5
