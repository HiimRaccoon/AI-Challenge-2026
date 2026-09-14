"""Deterministic corpus identity and shard-resume digests."""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Sequence

from .contracts import FrameRecord
from .corpus import resolve_keyframe


class ContentValidationMode(StrEnum):
    FAST = "fast"
    STRICT = "strict"


@dataclass(frozen=True)
class CorpusDigests:
    frames_jsonl_digest: str
    corpus_content_digest: str
    content_integrity_source: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_record(record: FrameRecord) -> bytes:
    payload = {
        "preprocess_run_id": record.preprocess_run_id,
        "video_id": record.video_id,
        "frame_id": record.frame_id,
        "keyframe_seq": record.keyframe_seq,
        "timestamp_ms": record.timestamp_ms,
        "keyframe_path": record.keyframe_path.replace("\\", "/"),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _record_digest(record: FrameRecord, data_root: Path, mode: ContentValidationMode) -> str:
    material = _canonical_record(record)
    if mode is ContentValidationMode.STRICT:
        material += b"\n" + sha256_bytes(resolve_keyframe(data_root, record).read_bytes()).encode("ascii")
    return sha256_bytes(material)


def _ordered_digest(record_digests: Sequence[str]) -> str:
    return sha256_bytes("\n".join(record_digests).encode("ascii"))


def compute_record_digests(
    corpus: Sequence[FrameRecord], data_root: Path, mode: ContentValidationMode,
    *, workers: int = 1,
) -> tuple[str, ...]:
    """Hash each record once, optionally parallelising strict image reads."""
    if workers <= 0:
        raise ValueError("digest workers must be positive")
    total = len(corpus)
    print(
        f"[wp03] validating {total:,} frame records "
        f"({mode.value}, workers={workers if mode is ContentValidationMode.STRICT else 1})",
        file=sys.stderr,
        flush=True,
    )
    digests: list[str] = []
    if workers == 1 or mode is ContentValidationMode.FAST:
        for completed, record in enumerate(corpus, start=1):
            digests.append(_record_digest(record, data_root, mode))
            if completed % 25_000 == 0 or completed == total:
                print(
                    f"[wp03] validation progress: {completed:,}/{total:,}",
                    file=sys.stderr,
                    flush=True,
                )
        return tuple(digests)

    # ThreadPoolExecutor.map eagerly submits its iterable on Python 3.12.
    # Submit moderate chunks so 714k frames do not create 714k Future objects.
    chunk_size = 256
    chunks = tuple(
        corpus[start : start + chunk_size]
        for start in range(0, total, chunk_size)
    )

    def digest_chunk(records: Sequence[FrameRecord]) -> tuple[str, ...]:
        return tuple(_record_digest(record, data_root, mode) for record in records)

    completed = 0
    next_report = 25_000
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wp03-digest") as executor:
        for chunk_digests in executor.map(digest_chunk, chunks):
            digests.extend(chunk_digests)
            completed += len(chunk_digests)
            if completed >= next_report or completed == total:
                print(
                    f"[wp03] validation progress: {completed:,}/{total:,}",
                    file=sys.stderr,
                    flush=True,
                )
                next_report += 25_000
    return tuple(digests)


def corpus_digests_from_records(
    record_digests: Sequence[str], *, mode: ContentValidationMode,
    frames_jsonl_digest: str, tv1_manifest_sha256: str | None,
    preprocess_run_id: str,
) -> CorpusDigests:
    """Assemble corpus identity from record digests already computed once."""
    if mode is ContentValidationMode.STRICT:
        source = "strict_image_bytes"
        material = list(record_digests)
    elif tv1_manifest_sha256:
        source = "tv1_manifest"
        material = [tv1_manifest_sha256, frames_jsonl_digest, *record_digests]
    else:
        source = "frames_jsonl_fallback"
        material = [preprocess_run_id, frames_jsonl_digest, *record_digests]
    return CorpusDigests(
        frames_jsonl_digest=frames_jsonl_digest,
        corpus_content_digest=_ordered_digest(material),
        content_integrity_source=source,
    )


def shard_digest_from_records(
    record_digests: Sequence[str], *, mode: ContentValidationMode,
    frames_jsonl_digest: str, tv1_manifest_sha256: str | None,
    preprocess_run_id: str,
) -> str:
    return corpus_digests_from_records(
        record_digests,
        mode=mode,
        frames_jsonl_digest=frames_jsonl_digest,
        tv1_manifest_sha256=tv1_manifest_sha256,
        preprocess_run_id=preprocess_run_id,
    ).corpus_content_digest


def compute_corpus_digests(
    corpus: Sequence[FrameRecord],
    data_root: Path,
    mode: ContentValidationMode,
    frames_jsonl_digest: str,
    tv1_manifest_sha256: str | None,
) -> CorpusDigests:
    """Compute the identity used to decide whether a build can be resumed."""

    record_digests = compute_record_digests(corpus, data_root, mode)
    return corpus_digests_from_records(
        record_digests,
        mode=mode,
        frames_jsonl_digest=frames_jsonl_digest,
        tv1_manifest_sha256=tv1_manifest_sha256,
        preprocess_run_id=corpus[0].preprocess_run_id,
    )


def compute_shard_input_digest(
    corpus_slice: Sequence[FrameRecord],
    data_root: Path,
    mode: ContentValidationMode,
    frames_jsonl_digest: str,
    tv1_manifest_sha256: str | None,
) -> str:
    """Return a shard-local digest using the same validation policy as corpus."""

    return compute_corpus_digests(
        corpus_slice, data_root, mode, frames_jsonl_digest, tv1_manifest_sha256
    ).corpus_content_digest
