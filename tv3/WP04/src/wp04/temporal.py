from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import FrameRecord


class TemporalResolver(Protocol):
    """WP04's only dependency on WP05 temporal data."""

    def frame_hypotheses(self, video_id: str, start_ms: int, end_ms: int) -> tuple[FrameRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class _Window:
    video_id: str
    frame_id: int
    start_ms: int
    end_ms: int


class LocalTemporalResolver:
    def __init__(self, frames: list[FrameRecord], windows: list[tuple[str, int, int, int]]):
        self._frames = frames
        self._windows = [_Window(*window) for window in windows]

    def frame_hypotheses(self, video_id: str, start_ms: int, end_ms: int) -> tuple[FrameRecord, ...]:
        midpoint = (start_ms + end_ms) // 2
        by_id = {frame.frame_id: frame for frame in self._frames if frame.video_id == video_id}
        matches = [window for window in self._windows if window.video_id == video_id and start_ms <= window.end_ms and end_ms >= window.start_ms]
        ranked = sorted(matches, key=lambda w: (-(min(end_ms, w.end_ms) - max(start_ms, w.start_ms)), abs(by_id[w.frame_id].timestamp_ms - midpoint), by_id[w.frame_id].timestamp_ms, w.frame_id))
        if ranked:
            return tuple(by_id[window.frame_id] for window in ranked if window.frame_id in by_id)
        candidates = tuple(by_id.values())
        return (min(candidates, key=lambda frame: (abs(frame.timestamp_ms - midpoint), frame.timestamp_ms, frame.frame_id)),) if candidates else ()
