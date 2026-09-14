"""TV4 factories for WP09's canonical exact-neighbor resolver.

TV1's current ``/frames/{video_id}`` endpoint hands over selected original
anchors, including their canonical frame ID and PTS.  It does *not* provide a
per-frame mapping.  WP09 therefore validates a selected anchor in the raw
stream and derives only bounded consecutive neighbors from that validated
anchor.  This adapter intentionally has no raw-PTS fallback.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote
from pathlib import Path

TV1_BASE_URL = os.environ.get("TV4_TV1_BASE_URL", "http://127.0.0.1:8000")


def _tv1_frames(video_id: str) -> list[dict]:
    url = f"{TV1_BASE_URL.rstrip('/')}/frames/{quote(video_id, safe='')}"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        from wp09.contracts import RefinementUnavailable

        raise RefinementUnavailable("anchor_not_found") from exc
    if not isinstance(payload, list):
        from wp09.contracts import RefinementUnavailable

        raise RefinementUnavailable("mapping_mismatch")
    return payload


def _tv1_media(video_id: str) -> dict:
    url = f"{TV1_BASE_URL.rstrip('/')}/media"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        from wp09.contracts import RefinementUnavailable
        raise RefinementUnavailable("media_identity_unproven") from exc
    matches = [row for row in rows if isinstance(row, dict) and row.get("video_id") == video_id]
    if len(matches) != 1:
        from wp09.contracts import RefinementUnavailable
        raise RefinementUnavailable("media_identity_unproven")
    return matches[0]


def resolver_for_request(request):
    """Build certified-run resolver from TV1 selected anchors/media authority."""
    from wp09.certification import load_run_certification
    from wp09.mapping import CanonicalAnchor, ExactFrameResolver, MediaIdentity, TrustedMediaValidator, InMemoryAnchorRegistry
    path = os.environ.get("TV4_EXACT_CERTIFICATION_PATH")
    if not path:
        from wp09.contracts import RefinementUnavailable
        raise RefinementUnavailable("run_certification_unproven")
    cert = load_run_certification(Path(path))
    media_row = _tv1_media(request.candidate.video_id)
    if media_row.get("preprocess_run_id") != request.context.preprocess_run_id:
        from wp09.contracts import RefinementUnavailable
        raise RefinementUnavailable("media_identity_mismatch")
    try:
        media = MediaIdentity(request.candidate.video_id, request.video_path, str(media_row["source_sha256"]), str(media_row["time_base"]), request.context)
        anchors = tuple(CanonicalAnchor(
            video_id=str(row["video_id"]), frame_id=int(row["frame_id"]), pts=int(row["pts"]),
            timestamp_ms=int(row["timestamp_ms"]), context=request.context, identity_guaranteed=True,
        ) for row in _tv1_frames(request.candidate.video_id))
    except (KeyError, TypeError, ValueError) as exc:
        from wp09.contracts import RefinementUnavailable
        raise RefinementUnavailable("anchor_not_found") from exc
    return ExactFrameResolver(_PyAVOriginalDecoder(), InMemoryAnchorRegistry(anchors), certification=cert, media=media, media_validator=_MEDIA_VALIDATOR)


def decoder_for_request(request):
    """Disable unproven sampling rather than returning raw decoder PTS."""
    from wp09.contracts import RefinementUnavailable

    raise RefinementUnavailable("canonical_identity_unproven")


class _PyAVOriginalDecoder:
    def duration_ms(self, video_path: Path) -> int:
        import av
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.duration is None or stream.time_base is None:
                raise ValueError("duration unavailable")
            return int(float(stream.duration * stream.time_base) * 1000)

    def raw_frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None):
        # Decode from a bounded seek position and retain only actual original
        # frames. No FPS sampling and no PTS-to-ID conversion occurs here.
        import av
        from wp09.mapping import RawDecodedFrame
        out = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise ValueError("time base unavailable")
            container.seek(max(0, int(start_ms / 1000 / float(stream.time_base))), stream=stream, any_frame=False, backward=True)
            for frame in container.decode(video=0):
                if frame.pts is None:
                    raise ValueError("PTS unavailable")
                timestamp_ms = int(float(frame.pts * stream.time_base) * 1000)
                if timestamp_ms < start_ms:
                    continue
                if timestamp_ms > end_ms:
                    break
                out.append(RawDecodedFrame(int(frame.pts), str(stream.time_base), timestamp_ms))
                if max_frames is not None and len(out) >= max_frames:
                    break
        return tuple(out)


_MEDIA_VALIDATOR = None
from wp09.mapping import TrustedMediaValidator
_MEDIA_VALIDATOR = TrustedMediaValidator()


def scorer_for_request(request):
    return None
