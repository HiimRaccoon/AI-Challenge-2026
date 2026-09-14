from __future__ import annotations

from pathlib import Path

from wp09.contracts import RefinementContext
from wp09.mapping import MappedVideoDecoder, RawDecodedFrame, ResolvedFrameIdentity


class RawDecoder:
    def duration_ms(self, video_path: Path) -> int:
        return 100

    def raw_frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None) -> tuple[RawDecodedFrame, ...]:
        return (RawDecodedFrame(pts=7, time_base="1/1000", timestamp_ms=19), RawDecodedFrame(pts=21, time_base="1/1000", timestamp_ms=57))


class Resolver:
    def resolve_frame(self, video_id: str, pts: int, time_base: str, context: RefinementContext) -> ResolvedFrameIdentity:
        return ResolvedFrameIdentity(frame_id={7: 3, 21: 8}[pts], timestamp_ms={7: 19, 21: 57}[pts], pts=pts)


def test_decoder_uses_mapping_resolver_for_true_original_frame_id() -> None:
    """Catches decoder index/FPS being substituted for TV1's canonical mapping."""

    decoder = MappedVideoDecoder(RawDecoder(), Resolver(), video_id="L21_V001", context=RefinementContext("run-1", "media/1", "map-vfr", "decoder-1", "model-1", "config-1"))

    frames = decoder.frames_between(Path("L21_V001.mp4"), 0, 100, 24)

    assert [frame.frame_id for frame in frames] == [3, 8]
    assert frames[1].timestamp_ms == 57


def test_mapping_identity_mismatch_is_a_fatal_mapping_failure() -> None:
    """Catches silently emitting a canonical frame whose PTS provenance disagrees with decode."""
    class BadResolver:
        def resolve_frame(self, video_id: str, pts: int, time_base: str, context: RefinementContext) -> ResolvedFrameIdentity:
            return ResolvedFrameIdentity(frame_id=1, timestamp_ms=999, pts=pts)

    from wp09.contracts import RefinementUnavailable
    decoder = MappedVideoDecoder(RawDecoder(), BadResolver(), video_id="L21_V001", context=RefinementContext("run-1", "media/1", "map-vfr", "decoder-1", "model-1", "config-1"))
    import pytest
    with pytest.raises(RefinementUnavailable, match="mapping_failure"):
        decoder.frames_between(Path("L21_V001.mp4"), 0, 100, 24)
