"""Read-only ingestion and validation of upstream TV1 preprocessing run reports and metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessingReport:
    run_id: str
    is_valid: bool
    status: str
    video_count: int = 0
    keyframe_count: int = 0
    thumbnail_count: int = 0
    storage_bytes: int = 0
    throughput_fps: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ingest_preprocessing_run_report(run_dir: Path, expected_run_id: str | None = None) -> PreprocessingReport:
    """Read-only ingestion of TV1 preprocessing run manifest. Never executes preprocessing commands."""
    if not run_dir.exists():
        return PreprocessingReport(
            run_id=expected_run_id or "unknown",
            is_valid=False,
            status="ACTUALLY MISSING",
            errors=(f"Preprocessing run directory does not exist: {run_dir}",),
        )

    # Inspect manifest
    manifest_path = run_dir / "manifest" / "corpus_manifest.json"
    if not manifest_path.exists():
        # check direct manifest.json
        manifest_path = run_dir / "manifest.json"

    if not manifest_path.exists():
        return PreprocessingReport(
            run_id=expected_run_id or run_dir.name,
            is_valid=False,
            status="INCOMPATIBLE",
            errors=(f"Missing required manifest file in {run_dir}",),
        )

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return PreprocessingReport(
            run_id=expected_run_id or run_dir.name,
            is_valid=False,
            status="INCOMPATIBLE",
            errors=(f"Failed to read/parse manifest: {exc}",),
        )

    if isinstance(data, list):
        video_count = len(data)
        run_id = expected_run_id or run_dir.name
        keyframe_count = sum(len(item.get("keyframes", [])) if isinstance(item, dict) else 0 for item in data)
        thumbnail_count = keyframe_count
        storage_bytes = 0
        throughput_fps = None
        provenance = {"manifest_type": "list", "video_count": video_count}
    elif isinstance(data, dict):
        run_id = data.get("preprocess_run_id", data.get("run_id", expected_run_id or run_dir.name))
        if expected_run_id and run_id != expected_run_id:
            return PreprocessingReport(
                run_id=run_id,
                is_valid=False,
                status="INCOMPATIBLE",
                errors=(f"Run ID mismatch: expected {expected_run_id}, got {run_id}",),
            )

        videos = data.get("videos", data.get("video_list", []))
        video_count = len(videos) if isinstance(videos, list) else int(data.get("unique_video_count", 0))

        # Read frames / keyframes / thumbnails metadata
        keyframe_count = int(data.get("keyframe_count", data.get("keyframes", 0)))
        thumbnail_count = int(data.get("thumbnail_count", data.get("thumbnails", 0)))

        # Estimate or read storage metrics
        storage_bytes = int(data.get("storage_bytes", data.get("total_bytes", 0)))
        throughput_fps = data.get("throughput_fps")
        if throughput_fps is not None:
            try:
                throughput_fps = float(throughput_fps)
            except (ValueError, TypeError):
                throughput_fps = None
        provenance = data.get("provenance", {})
    else:
        return PreprocessingReport(
            run_id=expected_run_id or run_dir.name,
            is_valid=False,
            status="INCOMPATIBLE",
            errors=("Manifest JSON is neither a list nor a dictionary",),
        )

    return PreprocessingReport(
        run_id=run_id,
        is_valid=True,
        status="READY",
        video_count=video_count,
        keyframe_count=keyframe_count,
        thumbnail_count=thumbnail_count,
        storage_bytes=storage_bytes,
        throughput_fps=throughput_fps,
        provenance=provenance,
    )
