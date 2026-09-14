from __future__ import annotations

from pathlib import Path

from wp09.decoder import DecodedFrame, sample_two_stage


class FakeDecoder:
    def duration_ms(self, video_path: Path) -> int:
        return 1_000

    def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float) -> tuple[DecodedFrame, ...]:
        del video_path, max_fps
        return tuple(
            DecodedFrame(frame_id=frame_id, pts=frame_id, time_base="1/100", timestamp_ms=frame_id * 10)
            for frame_id in range(101)
            if start_ms <= frame_id * 10 <= end_ms
        )


def test_sampler_clips_pts_window_and_resamples_around_best_coarse_frame() -> None:
    """Catches window underflow or dense sampling around the input rather than the best coarse frame."""

    result = sample_two_stage(
        decoder=FakeDecoder(),
        video_path=Path("L21_V001.mp4"),
        center_ms=50,
        radius_ms=600,
        coarse_sample_fps=2,
        dense_radius_ms=100,
        dense_sample_fps=24,
        rank_frame=lambda frame: float(frame.frame_id),
    )

    assert result.window.start_ms == 0
    assert result.window.end_ms == 650
    assert result.best_coarse_frame.frame_id == 65
    assert {frame.frame_id for frame in result.dense_frames} == set(range(55, 66))
