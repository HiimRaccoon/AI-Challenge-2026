"""Optional PyAV decoder that preserves upstream PTS-to-frame identities."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from .contracts import ContractError
from .decoder import DecodeBudgetExhausted
from .mapping import RawDecodedFrame


class PyAVVideoDecoder:
    """Decode raw original-video PTS frames; mapping belongs to MappedVideoDecoder."""

    def duration_ms(self, video_path: Path) -> int:
        av = _load_av()
        with av.open(str(video_path)) as container:
            stream = _video_stream(container)
            if stream.duration is not None and stream.time_base is not None:
                return _milliseconds(stream.duration, stream.time_base)
            if container.duration is not None:
                return int(container.duration / 1_000)
        raise ContractError("original video has no PTS duration")

    def raw_frames_between(
        self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None
    ) -> tuple[RawDecodedFrame, ...]:
        if start_ms < 0 or end_ms < start_ms or max_fps <= 0:
            raise ContractError("decoder bounds are invalid")
        av = _load_av()
        accepted: list[RawDecodedFrame] = []
        minimum_gap_ms = 1_000 / max_fps
        last_timestamp_ms: int | None = None
        with av.open(str(video_path)) as container:
            stream = _video_stream(container)
            if stream.time_base is None:
                raise ContractError("original video stream has no time_base")
            time_base = str(stream.time_base)
            # Seeking is approximate (typically the preceding keyframe), so
            # warm-up frames still count against max_frames below.
            container.seek(_pts_at_milliseconds(start_ms, stream.time_base), stream=stream, backward=True, any_frame=False)
            physical_frame_count = 0
            for frame in container.decode(stream):
                physical_frame_count += 1
                if frame.pts is None:
                    raise ContractError("original video frame has no PTS")
                timestamp_ms = _milliseconds(frame.pts, stream.time_base)
                if timestamp_ms < start_ms:
                    if max_frames is not None and physical_frame_count >= max_frames:
                        raise DecodeBudgetExhausted()
                    continue
                if timestamp_ms > end_ms:
                    break
                if last_timestamp_ms is not None and timestamp_ms - last_timestamp_ms < minimum_gap_ms:
                    if max_frames is not None and physical_frame_count >= max_frames:
                        break
                    continue
                accepted.append(
                    RawDecodedFrame(
                        pts=int(frame.pts),
                        time_base=time_base,
                        timestamp_ms=timestamp_ms,
                        image_rgb=frame.to_ndarray(format="rgb24"),
                    )
                )
                last_timestamp_ms = timestamp_ms
                # This bound covers pre-window warm-up and accepted output; a
                # keyframe far before the requested window cannot force an
                # unbounded decode from stream start.
                if max_frames is not None and physical_frame_count >= max_frames:
                    break
        return tuple(accepted)


def _load_av() -> object:
    try:
        import av
    except ImportError as exc:
        raise ContractError("PyAV is required for original-video decoding; install aic2026-wp09[gpu]") from exc
    return av


def _video_stream(container: object) -> object:
    streams = getattr(container, "streams").video
    if not streams:
        raise ContractError("original media contains no video stream")
    return streams[0]


def _milliseconds(pts: int, time_base: Fraction) -> int:
    return int(pts * time_base * 1_000)


def _pts_at_milliseconds(timestamp_ms: int, time_base: Fraction) -> int:
    return int(Fraction(timestamp_ms, 1_000) / time_base)
