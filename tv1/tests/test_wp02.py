"""Regression tests for WP02 retrieval-frame coverage policy."""

from __future__ import annotations

from pathlib import Path
import sys

import av
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wp02_shot_keyframe import (  # noqa: E402
    KeyframePolicy,
    ShotKeyframeExtractor,
    keep_after_dedup,
    process_single_video,
    selection_reason_for_frame,
)


def test_dense_defaults_are_safe_for_a_new_run() -> None:
    policy = KeyframePolicy.from_config({})

    assert policy.target_eval_fps == 10.0
    assert policy.min_gap_ms == 250
    assert policy.max_gap_ms == 1000


def test_max_gap_is_a_mandatory_support_frame() -> None:
    reason = selection_reason_for_frame(
        gap_ms=2000,
        boundary_detected=False,
        min_gap_ms=500,
        max_gap_ms=2000,
    )

    assert reason == "max_gap_support"
    assert keep_after_dedup(reason=reason, distance=0, threshold=12)


def test_boundary_frame_still_uses_perceptual_dedup() -> None:
    reason = selection_reason_for_frame(
        gap_ms=600,
        boundary_detected=True,
        min_gap_ms=500,
        max_gap_ms=2000,
    )

    assert reason == "boundary_guard"
    assert not keep_after_dedup(reason=reason, distance=12, threshold=12)
    assert keep_after_dedup(reason=reason, distance=13, threshold=12)


def test_no_selection_before_boundary_or_max_gap() -> None:
    assert selection_reason_for_frame(
        gap_ms=499,
        boundary_detected=True,
        min_gap_ms=500,
        max_gap_ms=2000,
    ) is None
    assert selection_reason_for_frame(
        gap_ms=1999,
        boundary_detected=False,
        min_gap_ms=500,
        max_gap_ms=2000,
    ) is None


@pytest.mark.parametrize(
    "config, message",
    [
        ({"min_gap_ms": 0}, "min_gap_ms"),
        ({"min_gap_ms": 2001, "max_gap_ms": 2000}, "max_gap_ms"),
        ({"target_eval_fps": 0.1, "max_gap_ms": 2000}, "target_eval_fps"),
        ({"jpeg_quality": 101}, "JPEG qualities"),
    ],
)
def test_invalid_policy_is_rejected(config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        KeyframePolicy.from_config(config)


def test_static_video_keeps_mandatory_support_frames(tmp_path: Path) -> None:
    """Identical pHashes must not create a long hole in the frame catalog."""
    video_path = tmp_path / "static.mp4"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("mpeg4", rate=25)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    for _ in range(75):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    run_dir = tmp_path / "run"
    result = process_single_video(
        "STATIC_V001",
        video_path,
        run_dir,
        "run-test-dense",
        {
            "target_eval_fps": 5.0,
            "min_gap_ms": 500,
            "max_gap_ms": 1000,
            "dedup_threshold": 64,
        },
    )

    assert result["status"] == "success"
    records = pd.read_parquet(run_dir / "mappings" / "STATIC_V001.parquet")
    assert len(records) >= 3
    assert "max_gap_support" in set(records["selection_reason"])
    assert records["timestamp_ms"].diff().dropna().max() <= 1000


def test_run_policy_prevents_mixing_sparse_and_dense_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = ShotKeyframeExtractor(
        tmp_path / "raw",
        run_dir,
        {"preprocess_run_id": "run-dense", "max_gap_ms": 2000},
    )
    first._prepare_run_policy()
    first._prepare_run_policy()

    policy = (run_dir / "preprocess_policy.json").read_text(encoding="utf-8")
    assert '"preprocess_run_id": "run-dense"' in policy

    changed = ShotKeyframeExtractor(
        tmp_path / "raw",
        run_dir,
        {"preprocess_run_id": "run-dense", "max_gap_ms": 3000},
    )
    with pytest.raises(RuntimeError, match="use a new run directory"):
        changed._prepare_run_policy()


def test_legacy_run_without_policy_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    mappings_dir = run_dir / "mappings"
    mappings_dir.mkdir(parents=True)
    (mappings_dir / "L21_V001.parquet").write_bytes(b"legacy")

    extractor = ShotKeyframeExtractor(
        tmp_path / "raw",
        run_dir,
        {"preprocess_run_id": "run-dense"},
    )
    with pytest.raises(RuntimeError, match="legacy mappings"):
        extractor._prepare_run_policy()
