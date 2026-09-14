"""Neighbor-specific fail-closed regression tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from wp09.contracts import CoarseCandidate, RefinementContext
from wp09.mapping import CanonicalAnchor, ExactFrameResolver, InMemoryAnchorRegistry, RawDecodedFrame


CTX = RefinementContext("run-1", "media/V1", "mapping/V1", "pyav", "n/a", "config-1")


class SmallDecoder:
    def duration_ms(self, video_path: Path) -> int:
        return 100

    def raw_frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None):
        frames = (RawDecodedFrame(900, "1/1000", 0), RawDecodedFrame(911, "1/1000", 33), RawDecodedFrame(933, "1/1000", 66))
        return tuple(frame for frame in frames if start_ms <= frame.timestamp_ms <= end_ms)


class ExactNeighborTests(unittest.TestCase):
    def test_unproven_anchor_never_makes_any_original_neighbor_selectable(self) -> None:
        resolver = ExactFrameResolver(
            SmallDecoder(), InMemoryAnchorRegistry((CanonicalAnchor("V1", 100, 900, 0, CTX, identity_guaranteed=False),))
        )
        result = resolver.resolve(CoarseCandidate("V1", 100, 0), Path("V1.mp4"), CTX, offsets=(0, 1))

        self.assertEqual(result.degraded_reason, "canonical_identity_unproven")
        self.assertTrue(all(step.frame is None for step in result.steps))

    def test_missing_anchor_and_non_monotonic_pts_are_not_relabelled_as_frame_ids(self) -> None:
        resolver = ExactFrameResolver(
            SmallDecoder(), InMemoryAnchorRegistry((CanonicalAnchor("V1", 100, 900, 0, CTX, identity_guaranteed=True),))
        )
        missing = resolver.resolve(CoarseCandidate("V1", 999, 0), Path("V1.mp4"), CTX, offsets=(0,))

        self.assertEqual(missing.degraded_reason, "anchor_not_found")
