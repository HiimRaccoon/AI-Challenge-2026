"""Shared checked worker implementation; model packages remain lazy imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..contracts import utc_now_iso8601


class EncoderAdapter(Protocol):
    def encode_images(self, paths: Sequence[Path]) -> np.ndarray: ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...


def _encode_in_batches(encode, values: Sequence[object], batch_size: int) -> np.ndarray:
    """Encode bounded chunks while preserving the request order."""

    if batch_size <= 0:
        raise ValueError("worker batch_size must be positive")
    batches = [encode(values[start : start + batch_size]) for start in range(0, len(values), batch_size)]
    if not batches:
        raise ValueError("worker request must contain at least one input")
    return np.concatenate(batches, axis=0)


def _error_type(exc: Exception) -> tuple[str, bool]:
    message = str(exc).lower()
    if "out of memory" in message or "cuda oom" in message:
        return "cuda_oom", True
    if "bfloat16" in message and ("unsupported" in message or "not implemented" in message):
        return "unsupported_bfloat16", True
    return "worker_error", False


def _normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError("encoder returned invalid embeddings")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder returned zero-norm embeddings")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _write_status(status_path: Path, payload: dict[str, object]) -> None:
    status_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def run_worker(adapter: EncoderAdapter, request_path: Path) -> int:
    """Execute exactly one request and always emit a status when its path is available."""

    raw = json.loads(request_path.read_text(encoding="utf-8"))
    status_path = Path(str(raw["status_path"]))
    base = {
        "schema_version": "1.0.0",
        "job_id": raw.get("job_id"),
        "request_sha256": raw.get("request_sha256"),
        "attempt": raw.get("attempt"),
        "finished_at_utc": utc_now_iso8601(),
    }
    try:
        operation = raw.get("operation")
        configure = getattr(adapter, "configure", None)
        if callable(configure):
            configure(dtype=str(raw["dtype"]))
        batch_size = raw.get("batch_size")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("worker request batch_size is invalid")
        if operation == "encode_images":
            values = _encode_in_batches(
                adapter.encode_images, tuple(Path(path) for path in raw.get("image_paths", ())), batch_size
            )
        elif operation == "encode_text":
            values = _encode_in_batches(adapter.encode_text, tuple(str(text) for text in raw.get("texts", ())), batch_size)
        else:
            _write_status(status_path, {**base, "status": "failed", "error_type": "unsupported_operation", "message": "unsupported operation", "retryable": False})
            return 2
        vectors = _normalize(values)
        output_path = Path(str(raw["output_path"]))
        with output_path.open("wb") as stream:
            np.save(stream, vectors, allow_pickle=False)
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        _write_status(
            status_path,
            {
                **base,
                "status": "ok",
                "output_sha256": digest,
                "count": int(vectors.shape[0]),
                "dimension": int(vectors.shape[1]),
                "shape": list(vectors.shape),
                "dtype": "float32",
                "normalized": True,
                "retryable": False,
            },
        )
        return 0
    except Exception as exc:  # the process boundary intentionally normalizes backend exceptions
        error_type, retryable = _error_type(exc)
        _write_status(status_path, {**base, "status": "failed", "error_type": error_type, "message": str(exc), "retryable": retryable})
        return 1
