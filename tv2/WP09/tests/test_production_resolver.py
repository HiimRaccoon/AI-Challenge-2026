"""Proof fixtures for bounded canonical original-frame resolution."""

from __future__ import annotations

from pathlib import Path
import unittest

from wp09.contracts import CoarseCandidate, RefinementContext
from wp09.mapping import (
    CanonicalAnchor,
    ExactFrameResolver,
    InMemoryAnchorRegistry,
    RawDecodedFrame,
)


CONTEXT = RefinementContext("run-1", "media/L21_V001", "mapping/L21_V001", "pyav", "n/a", "config-1")


class RawDecoder:
    """VFR-like decoded original stream: PTS is deliberately not frame_id."""

    def __init__(self, frames: tuple[RawDecodedFrame, ...] | None = None) -> None:
        self.frames = frames or (
            RawDecodedFrame(10_000, "1/1000", 0),
            RawDecodedFrame(10_041, "1/1000", 41),
            RawDecodedFrame(10_093, "1/1000", 93),
            RawDecodedFrame(10_126, "1/1000", 126),
            RawDecodedFrame(10_180, "1/1000", 180),
        )

    def duration_ms(self, video_path: Path) -> int:
        return 180

    def raw_frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None):
        assert max_fps == float("inf")  # Exact stepping must not apply nominal-FPS sampling.
        frames = tuple(frame for frame in self.frames if start_ms <= frame.timestamp_ms <= end_ms)
        return frames if max_frames is None else frames[:max_frames]


def _resolver(raw: RawDecoder | None = None) -> ExactFrameResolver:
    anchors = (
        CanonicalAnchor("L21_V001", 700, 10_000, 0, CONTEXT, identity_guaranteed=True),
        CanonicalAnchor("L21_V001", 703, 10_126, 126, CONTEXT, identity_guaranteed=True),
    )
    return ExactFrameResolver(raw or RawDecoder(), InMemoryAnchorRegistry(anchors))


class ProductionResolverTests(unittest.TestCase):
    def test_cross_anchor_agreement_never_promotes_offset_arithmetic_to_identity_proof(self) -> None:
        result = _resolver().resolve(
            CoarseCandidate("L21_V001", 700, 0), Path("L21_V001.mp4"), CONTEXT, offsets=(0, 1, 2, 3)
        )

        self.assertEqual(result.degraded_reason, "canonical_identity_unproven")
        self.assertTrue(all(step.frame is None for step in result.steps))

    def test_repeated_reverse_offsets_without_per_frame_authority_remain_non_selectable(self) -> None:
        result = _resolver().resolve(
            CoarseCandidate("L21_V001", 703, 126), Path("L21_V001.mp4"), CONTEXT, offsets=(0, -1, -2, -3)
        )

        self.assertEqual(result.degraded_reason, "canonical_identity_unproven")
        self.assertTrue(all(step.frame is None for step in result.steps))

    def test_first_and_last_original_frame_do_not_fabricate_neighbors_without_independent_authority(self) -> None:
        first = _resolver().resolve(CoarseCandidate("L21_V001", 700, 0), Path("L21_V001.mp4"), CONTEXT, offsets=(-1, 0))
        last = _resolver().resolve(CoarseCandidate("L21_V001", 703, 126), Path("L21_V001.mp4"), CONTEXT, offsets=(0, 2))

        self.assertEqual(first.steps[0].degraded_reason, "canonical_identity_unproven")
        self.assertIsNone(first.steps[0].frame)
        self.assertIsNone(first.steps[1].frame)
        self.assertEqual(last.steps[1].degraded_reason, "canonical_identity_unproven")
        self.assertIsNone(last.steps[1].frame)

    def test_anomalous_pts_and_stale_provenance_fail_closed(self) -> None:
        duplicate = RawDecoder((RawDecodedFrame(10_000, "1/1000", 0), RawDecodedFrame(10_000, "1/1000", 41)))
        duplicate_result = _resolver(duplicate).resolve(
            CoarseCandidate("L21_V001", 700, 0), Path("L21_V001.mp4"), CONTEXT, offsets=(0, 1)
        )
        stale = RefinementContext("stale-run", CONTEXT.media_record_ref, CONTEXT.mapping_ref, "pyav", "n/a", "config-1")
        stale_result = _resolver().resolve(CoarseCandidate("L21_V001", 700, 0), Path("L21_V001.mp4"), stale, offsets=(0,))

        self.assertEqual(duplicate_result.degraded_reason, "duplicate_pts")
        self.assertTrue(all(not step.frame or not step.frame.submission_selectable for step in duplicate_result.steps))
        self.assertEqual(stale_result.degraded_reason, "source_mismatch")

    def test_missing_and_non_monotonic_pts_fail_closed(self) -> None:
        missing = RawDecoder((RawDecodedFrame(None, "1/1000", 0),))
        non_monotonic = RawDecoder((RawDecodedFrame(10_000, "1/1000", 0), RawDecodedFrame(9_999, "1/1000", 41)))

        missing_result = _resolver(missing).resolve(CoarseCandidate("L21_V001", 700, 0), Path("L21_V001.mp4"), CONTEXT, offsets=(0,))
        non_monotonic_result = _resolver(non_monotonic).resolve(CoarseCandidate("L21_V001", 700, 0), Path("L21_V001.mp4"), CONTEXT, offsets=(0, 1))

        self.assertEqual(missing_result.degraded_reason, "pts_unavailable")
        self.assertEqual(non_monotonic_result.degraded_reason, "non_monotonic_pts")
