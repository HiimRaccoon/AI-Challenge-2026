"""TV1 corpus loading and filesystem-boundary validation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import ContractError, FrameRecord


def _safe_relative_path(raw_path: str, field: str) -> Path:
    windows_path = PureWindowsPath(raw_path)
    normalized = raw_path.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    if (
        not raw_path
        or raw_path.startswith(("/", "\\"))
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ContractError(f"{field} must be a safe relative path")
    return Path(*posix_path.parts)


@lru_cache(maxsize=16)
def _resolved_root(raw_root: str) -> Path:
    return Path(raw_root).resolve(strict=True)


@lru_cache(maxsize=4096)
def _resolved_keyframe_directory(raw_root: str, relative_parent: str) -> Path:
    root = _resolved_root(raw_root)
    directory = (root / relative_parent).resolve(strict=True)
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ContractError("keyframe_path resolves outside data_root") from exc
    if not directory.is_dir():
        raise ContractError("keyframe_path parent must be a directory")
    return directory


def resolve_keyframe(data_root: Path, record: FrameRecord) -> Path:
    """Resolve one TV1 keyframe, rejecting paths outside ``data_root``."""

    root = _resolved_root(str(data_root))
    relative_path = _safe_relative_path(record.keyframe_path, "keyframe_path")
    directory = _resolved_keyframe_directory(str(data_root), relative_path.parent.as_posix())
    candidate = directory / relative_path.name
    # Generated keyframes are regular files. Resolve the rare symlink itself
    # so the containment guarantee remains intact without paying 714k full
    # Path.resolve calls for an ordinary corpus.
    if candidate.is_symlink():
        candidate = candidate.resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ContractError("keyframe_path resolves outside data_root") from exc
    if not candidate.is_file():
        raise ContractError("keyframe_path must point to an existing file")
    return candidate


def _read_records(frames_file: Path) -> Iterable[FrameRecord]:
    if frames_file.suffix == ".parquet":
        try:
            rows = pq.read_table(frames_file).to_pylist()
        except (OSError, ValueError, pa.ArrowException) as exc:
            raise ContractError("frames.parquet cannot be read") from exc
        for row_number, payload in enumerate(rows, start=1):
            if not isinstance(payload, dict):
                raise ContractError(f"frames.parquet row {row_number} must be an object")
            yield FrameRecord.from_dict(payload)
        return
    if frames_file.suffix != ".jsonl":
        raise ContractError("frames must use .parquet or .jsonl")
    try:
        lines = frames_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError("frames path cannot be read") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"frames.jsonl line {line_number} is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ContractError(f"frames.jsonl line {line_number} must be an object")
        yield FrameRecord.from_dict(payload)


def load_corpus(
    data_root: Path,
    frames_path: PurePosixPath,
    selected_videos: frozenset[str] | None,
) -> tuple[FrameRecord, ...]:
    """Load, validate and canonically order TV1 frame records."""

    root = data_root.resolve(strict=True)
    relative_frames = _safe_relative_path(frames_path.as_posix(), "frames")
    frames_file = (root / relative_frames).resolve(strict=True)
    try:
        frames_file.relative_to(root)
    except ValueError as exc:
        raise ContractError("frames path resolves outside data_root") from exc

    corpus: list[FrameRecord] = []
    seen_keys: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    for record in _read_records(frames_file):
        if selected_videos is not None and record.video_id not in selected_videos:
            continue
        key = (record.video_id, record.frame_id)
        if key in seen_keys:
            raise ContractError("duplicate (video_id, frame_id) in frames corpus")
        seen_keys.add(key)
        run_ids.add(record.preprocess_run_id)
        resolve_keyframe(root, record)
        corpus.append(record)
    if not corpus:
        raise ContractError("frames corpus is empty after filtering")
    if len(run_ids) != 1:
        raise ContractError("frames corpus must have one preprocess_run_id")
    return tuple(sorted(corpus, key=lambda record: (record.video_id, record.frame_id)))
