"""Deterministic metadata records assembled from TV1 and observed WP04 evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .contracts import ASRSegment, FrameRecord, MetadataRecord, OCRDetection
from .temporal import TemporalResolver


_COLLECTION_ID = re.compile(r"^(L\d+)(?:_|$)", re.IGNORECASE)


def collection_metadata(video_id: str) -> dict[str, str]:
    """Expose the stable L-prefix as navigation metadata, never as topic.

    Videos in one L collection may contain mixed topics, so this intentionally
    avoids assigning a semantic label or moving media files on disk.
    """
    match = _COLLECTION_ID.match(video_id.strip())
    if match is None:
        return {}
    collection_id = match.group(1).upper()
    return {
        "collection_id": collection_id,
        "collection_label": f"Videos_{collection_id}",
    }


def _scalar_media_fields(media: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in media.items()
        if key != "caption" and isinstance(value, (str, int, float, bool))
    }


def build_metadata(
    preprocess_run_id: str, video_id: str, frames: Sequence[FrameRecord], media: Mapping[str, Any],
    ocr: Sequence[OCRDetection], asr: Sequence[ASRSegment], temporal: TemporalResolver,
) -> MetadataRecord:
    """Choose an original frame from evidence, then a TV1 midpoint frame as fallback."""
    video_frames = sorted(
        (frame for frame in frames if frame.video_id == video_id),
        key=lambda frame: (frame.timestamp_ms, frame.frame_id),
    )
    if not video_frames:
        raise ValueError(f"no TV1 frames for {video_id}")
    video_ocr = sorted(
        (item for item in ocr if item.video_id == video_id),
        key=lambda item: (item.timestamp_ms, item.frame_id, item.evidence_id),
    )
    video_asr = sorted(
        (item for item in asr if item.video_id == video_id),
        key=lambda item: (item.start_ms, item.end_ms, item.segment_id),
    )
    fields = _scalar_media_fields(media)
    fields.update(collection_metadata(video_id))
    evidence_refs: tuple[str, ...] = ()
    if video_ocr:
        anchor = next(frame for frame in video_frames if frame.frame_id == video_ocr[0].frame_id)
        fields["ocr_text"] = [item.raw_text for item in video_ocr]
        evidence_refs = tuple(item.evidence_id for item in video_ocr)
    elif video_asr and (hypotheses := temporal.frame_hypotheses(video_id, video_asr[0].start_ms, video_asr[0].end_ms)):
        anchor = hypotheses[0]
        fields["asr_text"] = [item.raw_text for item in video_asr]
        evidence_refs = tuple(item.segment_id for item in video_asr)
    else:
        duration = media.get("duration_ms")
        midpoint = int(duration) // 2 if isinstance(duration, (int, float)) else (
            video_frames[0].timestamp_ms + video_frames[-1].timestamp_ms
        ) // 2
        anchor = min(video_frames, key=lambda frame: (abs(frame.timestamp_ms - midpoint), frame.timestamp_ms, frame.frame_id))
    return MetadataRecord(
        preprocess_run_id, video_id, anchor.frame_id, anchor.timestamp_ms, fields,
        evidence_refs, f"metadata:{video_id}:{anchor.frame_id}",
    )
