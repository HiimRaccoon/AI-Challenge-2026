"""Crash-safe, versioned Parquet storage for WP04 modality artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import OCRDetection, WP04RunIdentity
from .status import ModalityStatus, should_skip


class SimulatedCrash(RuntimeError):
    """Test-only interruption point between durable data and status writes."""


def pair_key(video_id: str, frame_id: int) -> tuple[str, int]:
    return (video_id, frame_id)


class ArtifactStore:
    """Versioned WP04 artifacts, with status persisted only after shard durability."""

    def __init__(self, run_dir: Path, identity: WP04RunIdentity) -> None:
        self.run_dir = Path(run_dir)
        self.identity = identity
        self.root = self.run_dir / "wp04" / identity.wp04_artifact_set_id

    def _shard_path(self, modality: str, video_id: str) -> Path:
        return self.root / modality / f"{video_id}.parquet"

    @property
    def _status_path(self) -> Path:
        return self.root / "status" / "modality_status.parquet"

    @staticmethod
    def _row(record: object) -> dict[str, Any]:
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        if isinstance(record, dict):
            return dict(record)
        raise TypeError("artifact record must be a mapping or define to_dict()")

    @staticmethod
    def _empty_schema(modality: str | None = None) -> pa.Schema | None:
        schemas = {
            "ocr": pa.schema([
                ("preprocess_run_id", pa.string()),
                ("video_id", pa.string()),
                ("frame_id", pa.int64()),
                ("timestamp_ms", pa.int64()),
                ("raw_text", pa.string()),
                ("normalized_text", pa.string()),
                ("bbox_xyxy_norm", pa.list_(pa.float64())),
                ("confidence", pa.float64()),
                ("model_name", pa.string()),
                ("model_version", pa.string()),
                ("evidence_id", pa.string()),
            ]),
            "asr": pa.schema([
                ("preprocess_run_id", pa.string()),
                ("video_id", pa.string()),
                ("segment_id", pa.string()),
                ("start_ms", pa.int64()),
                ("end_ms", pa.int64()),
                ("raw_text", pa.string()),
                ("normalized_text", pa.string()),
                ("confidence", pa.float64()),
                ("model_name", pa.string()),
                ("model_version", pa.string()),
            ]),
            "object": pa.schema([
                ("preprocess_run_id", pa.string()),
                ("video_id", pa.string()),
                ("frame_id", pa.int64()),
                ("timestamp_ms", pa.int64()),
                ("label", pa.string()),
                ("bbox_xyxy_norm", pa.list_(pa.float64())),
                ("confidence", pa.float64()),
                ("model_name", pa.string()),
                ("model_version", pa.string()),
                ("evidence_id", pa.string()),
            ]),
            "metadata": pa.schema([
                ("preprocess_run_id", pa.string()),
                ("video_id", pa.string()),
                ("frame_id", pa.int64()),
                ("timestamp_ms", pa.int64()),
                ("fields_json", pa.string()),
                ("evidence_refs", pa.list_(pa.string())),
                ("record_id", pa.string()),
            ]),
        }
        return schemas.get(str(modality)) if modality else None

    @staticmethod
    def _write_table(path: Path, rows: Iterable[dict[str, Any]], modality: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        materialized = []
        for row in rows:
            encoded = dict(row)
            if isinstance(encoded.get("fields"), dict):
                encoded["fields_json"] = json.dumps(encoded.pop("fields"), ensure_ascii=False, sort_keys=True)
            materialized.append(encoded)
        table = pa.Table.from_pylist(materialized, schema=ArtifactStore._empty_schema(modality) if not materialized else None)
        temporary = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
        pq.write_table(table, temporary)
        # Windows scanners can briefly hold either side of the rename after
        # parquet has been written. Retry the atomic replacement for a short
        # window instead of aborting a long-running preprocess job.
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)

    @staticmethod
    @contextmanager
    def _exclusive_lock(path: Path):
        """Cross-process one-byte file lock for status and promotion transactions."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except ImportError:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            try:
                yield
            finally:
                unlock()

    @staticmethod
    def _read_table(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        # Keep the parquet input handle scoped to this block. On Windows,
        # replacing a status file immediately after reading it can otherwise
        # fail with WinError 5 while the reader still owns the file handle.
        with pa.memory_map(str(path), "r") as source:
            table = pq.read_table(source)
            try:
                rows = table.to_pylist()
            finally:
                del table
        for row in rows:
            if "fields_json" in row:
                row["fields"] = json.loads(row.pop("fields_json"))
        return rows

    def write_records(self, modality: str, video_id: str, records: Sequence[object]) -> None:
        self._write_table(self._shard_path(modality, video_id), (self._row(row) for row in records), modality)

    def read_records(self, modality: str, video_id: str) -> list[dict[str, Any]]:
        return self._read_table(self._shard_path(modality, video_id))

    def read_all_records(self, modality: str) -> list[dict[str, Any]]:
        directory = self.root / modality
        if not directory.exists():
            return []
        rows: list[dict[str, Any]] = []
        for shard in sorted(directory.glob("*.parquet")):
            rows.extend(self._read_table(shard))
        return rows

    def _all_statuses(self) -> list[ModalityStatus]:
        rows = self._read_table(self._status_path)
        return [
            ModalityStatus(
                video_id=row["video_id"], modality=row["modality"],
                fingerprint=row["fingerprint"], state=row["state"],
                error_message=row.get("error_message"),
                preprocess_run_id=row.get("preprocess_run_id"),
                wp04_artifact_set_id=row.get("wp04_artifact_set_id"),
            )
            for row in rows
        ]

    def append_status(self, status: ModalityStatus) -> None:
        with self._exclusive_lock(self.root / "status" / ".modality_status.lock"):
            prior = [
                item for item in self._all_statuses()
                if (item.video_id, item.modality) != (status.video_id, status.modality)
            ]
            rows = [item.to_dict() for item in prior]
            rows.append({
                **status.to_dict(),
                "preprocess_run_id": status.preprocess_run_id or self.identity.preprocess_run_id,
                "wp04_artifact_set_id": status.wp04_artifact_set_id or self.identity.wp04_artifact_set_id,
            })
            self._write_table(self._status_path, rows)

    def status_for(self, video_id: str, modality: str) -> ModalityStatus | None:
        return next(
            (item for item in self._all_statuses() if (item.video_id, item.modality) == (video_id, modality)),
            None,
        )

    def completed(self, video_id: str, modality: str, fingerprint: str) -> bool:
        status = self.status_for(video_id, modality)
        if not should_skip(status, fingerprint):
            return False
        return status is not None and (
            status.state == "no_audio" or self._shard_path(modality, video_id).exists()
        )

    def commit_video(
        self, modality: str, video_id: str, records: Sequence[object], fingerprint: str,
        *, crash_before_status: bool = False,
    ) -> None:
        self.write_records(modality, video_id, records)
        if crash_before_status:
            raise SimulatedCrash("interrupted after durable shard write")
        self.append_status(ModalityStatus.ready(video_id, modality, fingerprint))

    def validate_ocr(
        self, records: Sequence[OCRDetection], frames: set[tuple[str, int]],
    ) -> None:
        invalid = [record for record in records if pair_key(record.video_id, record.frame_id) not in frames]
        if invalid:
            raise ValueError("OCR references a frame outside its video")

    def promote(self, modalities: Sequence[str]) -> dict[str, Path]:
        """Compact a validated versioned set into the stable canonical paths."""
        with self._exclusive_lock(self.run_dir / ".wp04-promotion.lock"):
            report_path = self.root / "reports" / "wp04-validation.json"
            if not report_path.exists():
                raise ValueError("promotion requires a validation report")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not report.get("valid"):
                raise ValueError("promotion requires a valid artifact set")
            promoted: dict[str, Path] = {}
            for modality in modalities:
                destination = self.run_dir / modality / f"{modality}.parquet"
                self._write_table(destination, self.read_all_records(modality), modality)
                promoted[modality] = destination
            return promoted
