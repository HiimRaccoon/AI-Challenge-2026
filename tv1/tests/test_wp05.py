"""
Tests for WP05 — Temporal Adjacency
"""
from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wp05_temporal_adj import TemporalAdjacency, lookup_by_timestamp, get_neighbors


class TestTemporalAdjacency:
    def _create_test_data(self, tmp_path):
        """Create minimal frames.parquet and media.parquet for testing."""
        frames_data = [
            {"schema_version": "1.0.0", "preprocess_run_id": "test", "video_id": "V001",
             "frame_id": 0, "keyframe_seq": 0, "timestamp_ms": 0, "pts": 0,
             "shot_id": "V001_shot_000", "keyframe_path": "kf/V001/0.jpg",
             "thumbnail_path": None, "selection_reason": "shot_representative",
             "sharpness_score": 100.0, "blur_score": 0.01, "created_at_utc": "2026-01-01T00:00:00Z"},
            {"schema_version": "1.0.0", "preprocess_run_id": "test", "video_id": "V001",
             "frame_id": 30, "keyframe_seq": 1, "timestamp_ms": 1000, "pts": 30000,
             "shot_id": "V001_shot_000", "keyframe_path": "kf/V001/1.jpg",
             "thumbnail_path": None, "selection_reason": "max_gap",
             "sharpness_score": 120.0, "blur_score": 0.008, "created_at_utc": "2026-01-01T00:00:00Z"},
            {"schema_version": "1.0.0", "preprocess_run_id": "test", "video_id": "V001",
             "frame_id": 90, "keyframe_seq": 2, "timestamp_ms": 3000, "pts": 90000,
             "shot_id": "V001_shot_001", "keyframe_path": "kf/V001/2.jpg",
             "thumbnail_path": None, "selection_reason": "shot_representative",
             "sharpness_score": 90.0, "blur_score": 0.011, "created_at_utc": "2026-01-01T00:00:00Z"},
        ]
        frames_df = pd.DataFrame(frames_data)
        frames_df.to_parquet(tmp_path / "frames.parquet")

        media_data = [
            {"video_id": "V001", "duration_ms": 5000, "width_px": 1920, "height_px": 1080},
        ]
        media_df = pd.DataFrame(media_data)
        (tmp_path / "media").mkdir(parents=True, exist_ok=True)
        media_df.to_parquet(tmp_path / "media" / "media.parquet")

        return tmp_path

    def test_build_temporal_index(self, tmp_path):
        run_dir = self._create_test_data(tmp_path)
        adj = TemporalAdjacency(run_dir)
        adj.run()

        output = run_dir / "temporal" / "temporal_frames.parquet"
        assert output.exists()

        df = pd.read_parquet(output)
        assert len(df) == 3
        assert "prev_frame_id" in df.columns
        assert "next_frame_id" in df.columns
        assert "window_start_ms" in df.columns
        assert "window_end_ms" in df.columns
        assert "timestamp_ms" in df.columns

    def test_prev_next_boundaries(self, tmp_path):
        run_dir = self._create_test_data(tmp_path)
        adj = TemporalAdjacency(run_dir)
        adj.run()

        df = pd.read_parquet(run_dir / "temporal" / "temporal_frames.parquet")
        df = df.sort_values("timestamp_ms").reset_index(drop=True)

        # First frame should have no prev
        first = df.iloc[0]
        assert pd.isna(first["prev_frame_id"]) or first.get("prev_frame_id") is None

        # Last frame should have no next
        last = df.iloc[-1]
        assert pd.isna(last["next_frame_id"]) or last.get("next_frame_id") is None

    def test_window_start_zero_for_first(self, tmp_path):
        run_dir = self._create_test_data(tmp_path)
        adj = TemporalAdjacency(run_dir)
        adj.run()

        df = pd.read_parquet(run_dir / "temporal" / "temporal_frames.parquet")
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        assert df.iloc[0]["window_start_ms"] == 0

    def test_timestamp_preserved(self, tmp_path):
        run_dir = self._create_test_data(tmp_path)
        adj = TemporalAdjacency(run_dir)
        adj.run()

        df = pd.read_parquet(run_dir / "temporal" / "temporal_frames.parquet")
        timestamps = sorted(df["timestamp_ms"].tolist())
        assert timestamps == [0, 1000, 3000]


class TestLookupFunctions:
    def _make_df(self):
        data = [
            {"video_id": "V001", "frame_id": 0, "timestamp_ms": 0,
             "window_start_ms": 0, "window_end_ms": 500},
            {"video_id": "V001", "frame_id": 30, "timestamp_ms": 1000,
             "window_start_ms": 500, "window_end_ms": 2000},
            {"video_id": "V001", "frame_id": 90, "timestamp_ms": 3000,
             "window_start_ms": 2000, "window_end_ms": 5000},
        ]
        return pd.DataFrame(data)

    def test_lookup_middle(self):
        df = self._make_df()
        result = lookup_by_timestamp(df, "V001", 1500)
        assert result is not None
        assert result["frame_id"] == 30

    def test_lookup_start(self):
        df = self._make_df()
        result = lookup_by_timestamp(df, "V001", 100)
        assert result is not None
        assert result["frame_id"] == 0

    def test_lookup_invalid_video(self):
        df = self._make_df()
        result = lookup_by_timestamp(df, "INVALID", 1000)
        assert result is None

    def test_get_neighbors(self):
        df = self._make_df()
        neighbors = get_neighbors(df, "V001", 30, 1500)
        assert len(neighbors) >= 2  # should include frame 0 and 30 at least
