"""PTS-preserving original-video decoding and bounded two-stage sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable, Literal, Protocol

from .contracts import ContractError, DecodeBudget


class DecodeBudgetExhausted(RuntimeError):
    """Physical decoder budget ended before a frame in the requested window."""


@dataclass(frozen=True)
class DecodedFrame:
    frame_id: int
    pts: int
    time_base: str
    timestamp_ms: int
    image_rgb: object | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.pts < 0 or self.timestamp_ms < 0:
            raise ContractError("decoded frame identifiers must be non-negative")
        if not self.time_base:
            raise ContractError("decoded frame time_base must be non-empty")


@dataclass(frozen=True)
class LocalWindow:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ContractError("local window bounds are invalid")


@dataclass(frozen=True)
class SamplingResult:
    window: LocalWindow
    coarse_frames: tuple[DecodedFrame, ...]
    best_coarse_frame: DecodedFrame
    dense_frames: tuple[DecodedFrame, ...]
    budget_exhausted: bool = False
    cache_hit: bool = False


class VideoDecoder(Protocol):
    def duration_ms(self, video_path: Path) -> int: ...
    def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None) -> tuple[DecodedFrame, ...]: ...


class MappingGuaranteedDecoder(VideoDecoder, Protocol):
    """Explicit adapter contract for decoders that already resolve TV1 mapping."""
    mapping_guaranteed: Literal[True]


def sample_two_stage(
    *, decoder: VideoDecoder, video_path: Path, center_ms: int, radius_ms: int,
    coarse_sample_fps: float, dense_radius_ms: int, dense_sample_fps: float,
    rank_frame: Callable[[DecodedFrame], float], budget: DecodeBudget | None = None,
    dense_seed_count: int = 1,
) -> SamplingResult:
    """Clip a PTS-derived window, sample coarsely, then densify promising regions.

    The budget limits the *frames handed to scoring*. Backends remain responsible
    for stopping physical decode efficiently, which keeps this protocol usable for
    both PyAV and TV1's decoder adapters.
    """
    if center_ms < 0 or radius_ms <= 0 or dense_radius_ms <= 0 or dense_seed_count <= 0:
        raise ContractError("sampling time bounds are invalid")
    if coarse_sample_fps <= 0 or dense_sample_fps <= 0:
        raise ContractError("sampling fps values must be positive")
    started = monotonic()
    duration_ms = decoder.duration_ms(video_path)
    if duration_ms <= 0:
        raise ContractError("original video duration must be positive")
    effective_radius = min(radius_ms, budget.max_window_ms // 2) if budget else radius_ms
    window = LocalWindow(max(0, center_ms - effective_radius), min(duration_ms, center_ms + effective_radius))
    coarse_limit = budget.max_decoded_frames if budget else None
    coarse = _decode_limited(decoder, video_path, window.start_ms, window.end_ms, coarse_sample_fps, coarse_limit)
    if not coarse:
        raise ContractError("coarse sampling returned no original-video frames")
    ranked_coarse = sorted(coarse, key=lambda f: (-rank_frame(f), f.timestamp_ms, f.frame_id))
    best = ranked_coarse[0]
    seeds = ranked_coarse[: min(dense_seed_count, budget.max_dense_regions if budget else dense_seed_count)]
    dense: list[DecodedFrame] = []
    exhausted = False
    for seed in seeds:
        if budget and budget.max_decode_time_ms is not None and (monotonic() - started) * 1_000 > budget.max_decode_time_ms:
            exhausted = True
            break
        start = max(window.start_ms, seed.timestamp_ms - dense_radius_ms)
        end = min(window.end_ms, seed.timestamp_ms + dense_radius_ms)
        remaining = None if budget is None else budget.max_decoded_frames - len(coarse) - len(dense)
        if remaining is not None and remaining <= 0:
            exhausted = True
            break
        dense.extend(_decode_limited(decoder, video_path, start, end, dense_sample_fps, remaining))
        if budget and budget.max_decode_time_ms is not None and (monotonic() - started) * 1_000 > budget.max_decode_time_ms:
            exhausted = True
            break
    if not dense:
        # The coarse best is already a canonical decoded frame and is a valid,
        # explicitly partial fallback when the execution budget is exhausted.
        if budget is not None:
            dense = [best]
            exhausted = True
        else:
            raise ContractError("dense sampling returned no original-video frames")
    # Mapping identity is frame_id. Dense frames remain distinct from coarse
    # frames: policies score the high-resolution local evidence only.
    unique = {frame.frame_id: frame for frame in dense}
    merged = tuple(sorted(unique.values(), key=lambda f: (f.timestamp_ms, f.frame_id)))
    if budget and len(coarse) + len(merged) > budget.max_decoded_frames:
        exhausted = True
        # Keep the best coarse frame and then closest original frames. This makes
        # a partial result useful instead of arbitrarily truncating in decode order.
        allowance = max(1, budget.max_decoded_frames - min(len(coarse), budget.max_decoded_frames - 1))
        ordered = sorted(merged, key=lambda f: (0 if f.frame_id == best.frame_id else 1, abs(f.timestamp_ms - best.timestamp_ms), f.timestamp_ms, f.frame_id))
        merged = tuple(sorted(ordered[:allowance], key=lambda f: (f.timestamp_ms, f.frame_id)))
    return SamplingResult(window, tuple(coarse), best, merged, exhausted)


def _decode_limited(decoder: VideoDecoder, video_path: Path, start_ms: int, end_ms: int, fps: float, max_frames: int | None) -> tuple[DecodedFrame, ...]:
    """Keep legacy unbounded test adapters usable; budgets require the strict API."""
    if max_frames is None:
        return decoder.frames_between(video_path, start_ms, end_ms, fps)
    return decoder.frames_between(video_path, start_ms, end_ms, fps, max_frames)
