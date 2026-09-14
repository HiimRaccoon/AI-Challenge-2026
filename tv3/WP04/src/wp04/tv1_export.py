"""Convert legacy TV1 Parquet artifacts into WP04 CLI input manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import AudioRecord, FrameRecord


class TV1ManifestExportError(ValueError):
    """Raised when a TV1 artifact set cannot be safely handed to WP04."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_tv1_path(value: object, run_dir: Path, field: str, video_id: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TV1ManifestExportError(f"TV1 {field} is missing for {video_id}")
    supplied = Path(value)
    candidates = (supplied,) if supplied.is_absolute() else tuple(base / supplied for base in (run_dir, *run_dir.parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise TV1ManifestExportError(f"TV1 {field} does not exist for {video_id}: {value}")


def _required(row: dict[str, Any], field: str, artifact: str) -> Any:
    value = row.get(field)
    if value is None:
        raise TV1ManifestExportError(f"TV1 {artifact} row is missing {field}")
    return value


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_tv1_manifests(run_dir: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Export TV1 ``frames.parquet`` and ``media/audio.parquet`` for WP04."""
    run_dir = run_dir.resolve()
    frames_path, audio_path = run_dir / "frames.parquet", run_dir / "media" / "audio.parquet"
    if not frames_path.is_file():
        raise TV1ManifestExportError(f"TV1 frames artifact is missing: {frames_path}")
    if not audio_path.is_file():
        raise TV1ManifestExportError(f"TV1 audio artifact is missing: {audio_path}")

    frames: list[dict[str, Any]] = []
    frame_video_ids: set[str] = set()
    for row in pq.read_table(frames_path).to_pylist():
        video_id = str(_required(row, "video_id", "frame"))
        keyframe_path = _resolve_tv1_path(row.get("keyframe_path"), run_dir, "keyframe_path", video_id)
        frame = FrameRecord(
            str(_required(row, "preprocess_run_id", "frame")), video_id,
            int(_required(row, "frame_id", "frame")), int(_required(row, "keyframe_seq", "frame")),
            int(_required(row, "timestamp_ms", "frame")), str(keyframe_path), _sha256(keyframe_path),
            str(row["shot_id"]) if row.get("shot_id") is not None else None,
            str(row["selection_reason"]) if row.get("selection_reason") is not None else None,
        )
        frame_video_ids.add(video_id)
        frames.append({
            "preprocess_run_id": frame.preprocess_run_id,
            "video_id": frame.video_id,
            "frame_id": frame.frame_id,
            "keyframe_seq": frame.keyframe_seq,
            "timestamp_ms": frame.timestamp_ms,
            "keyframe_path": frame.keyframe_path,
            "keyframe_sha256": frame.keyframe_sha256,
            "shot_id": frame.shot_id,
            "selection_reason": frame.selection_reason,
        })

    audio: list[dict[str, Any]] = []
    audio_video_ids: set[str] = set()
    for row in pq.read_table(audio_path).to_pylist():
        video_id = str(_required(row, "video_id", "audio"))
        if video_id in audio_video_ids:
            raise TV1ManifestExportError(f"TV1 audio has duplicate video_id: {video_id}")
        audio_video_ids.add(video_id)
        status = str(_required(row, "status", "audio"))
        declared_present = status == "ready"
        if status not in {"ready", "no_audio"}:
            raise TV1ManifestExportError(
                f"TV1 audio status for {video_id} is {status!r}; resolve it before exporting"
            )
        resolved_audio_path = (
            str(_resolve_tv1_path(row.get("audio_path"), run_dir, "audio_path", video_id)) if declared_present else None
        )
        checksum = str(_required(row, "audio_sha256", "audio")) if declared_present else None
        record = AudioRecord(
            str(_required(row, "preprocess_run_id", "audio")), video_id, resolved_audio_path,
            checksum, declared_present,
            int(row["duration_ms"]) if row.get("duration_ms") is not None else None,
        )
        audio.append({
            "preprocess_run_id": record.preprocess_run_id,
            "video_id": record.video_id,
            "audio_path": record.audio_path,
            "checksum": record.checksum,
            "declared_present": record.declared_present,
            "duration_ms": record.duration_ms,
        })

    missing_audio = sorted(frame_video_ids - audio_video_ids)
    if missing_audio:
        raise TV1ManifestExportError(f"TV1 audio artifact is missing records for: {', '.join(missing_audio)}")
    if not frames:
        raise TV1ManifestExportError("TV1 frames artifact contains no records")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_output, audio_output = output_dir / "frames.json", output_dir / "audio.json"
    if not overwrite:
        existing = [str(path) for path in (frames_output, audio_output) if path.exists()]
        if existing:
            raise TV1ManifestExportError(f"refusing to overwrite existing manifest(s): {', '.join(existing)}")
    _write_json(frames_output, frames)
    _write_json(audio_output, audio)
    return {
        "frames_json": str(frames_output),
        "audio_json": str(audio_output),
        "frame_count": len(frames),
        "audio_count": len(audio),
        "preprocess_run_ids": sorted({row["preprocess_run_id"] for row in frames}),
    }
