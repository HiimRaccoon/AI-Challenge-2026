"""Authoritative original-media lookup for exact-frame requests.

This module deliberately has no filename-derived fallback.  The registry is
small metadata supplied by WP00/WP01 and is read-only at runtime; it is not a
frame mapping or a preprocessing artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping


_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class MediaRecord:
    video_id: str
    original_video_path: str
    source_sha256: str
    preprocess_run_id: str
    time_base: str
    media_record_ref: str
    mapping_ref: str


def resolve_original_media_path(
    media_root: Path, video_id: str, registry: Mapping[str, MediaRecord] | None, preprocess_run_id: str,
    allowed_extensions: Iterable[str] | None = None,
) -> Path:
    """Resolve one read-only registry-owned original video, fail closed."""
    if not _VIDEO_ID.fullmatch(video_id):
        raise ValueError("invalid video_id")
    if registry is None or video_id not in registry:
        raise ValueError("authoritative media record unavailable")
    record = registry[video_id]
    if record.video_id != video_id or record.preprocess_run_id != preprocess_run_id:
        raise ValueError("media provenance mismatch")
    try:
        root = media_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("configured original media root is unavailable") from exc
    if not root.is_dir():
        raise ValueError("configured original media root is not a directory")
    raw_path = Path(record.original_video_path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError("original media path is not a safe relative path")
    try:
        candidate = (root / raw_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("original media file is unavailable") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("original media path escapes configured root") from exc
    if not candidate.is_file():
        raise ValueError("original media file is unavailable")
    allowed = {extension.lower() for extension in allowed_extensions or ()}
    if not allowed or candidate.suffix.lower() not in allowed:
        raise ValueError("original media extension is not allowed")
    return candidate


def resolve_derivative_image_path(
    asset_root: Path, registry_path: str, allowed_extensions: Iterable[str] | None,
) -> Path:
    """Resolve a server-supplied derivative only when it remains in its root."""
    try:
        root = asset_root.resolve(strict=True)
        raw_path = Path(registry_path)
        candidate = raw_path.resolve(strict=True) if raw_path.is_absolute() else (root / raw_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("derivative image is unavailable") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("derivative image escapes configured root") from exc
    allowed = {extension.lower() for extension in allowed_extensions or ()}
    if not candidate.is_file() or not allowed or candidate.suffix.lower() not in allowed:
        raise ValueError("derivative image is not allowed")
    return candidate
