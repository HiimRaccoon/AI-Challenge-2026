"""Checked JSON protocol between the orchestrator and model workers."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .contracts import ContractError, utc_now_iso8601

if TYPE_CHECKING:
    from .config import ModelRuntimeSpec


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compatibility_fingerprint(
    *,
    spec: "ModelRuntimeSpec | None" = None,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    adapter_hash: str = "",
) -> str:
    """Hash semantic settings only; compute dtype/device do not affect compatibility."""

    if spec is not None:
        payload = spec.semantic_payload()
    else:
        if revision is None or tokenizer_revision is None:
            raise ContractError("revision and tokenizer_revision are required without a model spec")
        payload = {"revision": revision, "tokenizer_revision": tokenizer_revision}
    payload["adapter_hash"] = adapter_hash
    return _canonical_sha256(payload)


def runtime_fingerprint(*, compatibility: str, dtype: str, device: str = "cuda") -> str:
    return _canonical_sha256({"compatibility": compatibility, "dtype": dtype, "device": device})


@dataclass(frozen=True)
class WorkerRequest:
    job_id: str
    operation: str
    model_key: str
    revision: str
    device: str
    dtype: str
    batch_size: int
    attempt: int
    image_paths: tuple[str, ...]
    texts: tuple[str, ...]
    request_path: Path
    output_path: Path
    status_path: Path
    sha256: str

    @classmethod
    def create(
        cls,
        job_dir: Path,
        operation: str,
        model_key: str,
        revision: str,
        device: str,
        dtype: str,
        batch_size: int,
        attempt: int,
        image_paths: Sequence[Path],
        texts: Sequence[str],
    ) -> "WorkerRequest":
        if not job_dir.is_dir():
            raise ContractError("job_dir must exist")
        if batch_size <= 0:
            raise ContractError("batch_size must be positive")
        if attempt <= 0:
            raise ContractError("attempt must be positive")
        if operation not in {"encode_images", "encode_text"}:
            raise ContractError("operation is unsupported")
        job_id = str(uuid.uuid4())
        request_path = job_dir / f"{job_id}.request.json"
        output_path = job_dir / f"{job_id}.output.npy.tmp"
        status_path = job_dir / f"{job_id}.status.json"
        body: dict[str, object] = {
            "schema_version": "1.0.0",
            "job_id": job_id,
            "operation": operation,
            "model_key": model_key,
            "revision": revision,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "attempt": attempt,
            "image_paths": [str(path) for path in image_paths],
            "texts": list(texts),
            "output_path": str(output_path),
            "status_path": str(status_path),
        }
        digest = _canonical_sha256(body)
        return cls(
            job_id=job_id,
            operation=operation,
            model_key=model_key,
            revision=revision,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            attempt=attempt,
            image_paths=tuple(str(path) for path in image_paths),
            texts=tuple(texts),
            request_path=request_path,
            output_path=output_path,
            status_path=status_path,
            sha256=digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "job_id": self.job_id,
            "request_sha256": self.sha256,
            "operation": self.operation,
            "model_key": self.model_key,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "attempt": self.attempt,
            "image_paths": list(self.image_paths),
            "texts": list(self.texts),
            "output_path": str(self.output_path),
            "status_path": str(self.status_path),
        }

    def write(self) -> None:
        self.request_path.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class WorkerStatus:
    status: str
    job_id: str
    request_sha256: str
    error_type: str | None
    message: str | None
    retryable: bool
    count: int | None
    dimension: int | None
    dtype: str | None
    normalized: bool | None
    output_sha256: str | None

    @classmethod
    def from_json(cls, path: Path, request: WorkerRequest) -> "WorkerStatus":
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("worker status cannot be read") from exc
        if payload.get("job_id") != request.job_id:
            raise ContractError("worker status job_id does not match request")
        if payload.get("request_sha256") != request.sha256:
            raise ContractError("worker status request_sha256 does not match request")
        status = payload.get("status")
        if status not in {"ok", "failed"}:
            raise ContractError("worker status is invalid")
        count = payload.get("count")
        dimension = payload.get("dimension")
        dtype = payload.get("dtype")
        normalized = payload.get("normalized")
        output_sha256 = payload.get("output_sha256")
        if status == "ok":
            if not isinstance(count, int) or count <= 0:
                raise ContractError("worker status count is invalid")
            if not isinstance(dimension, int) or dimension <= 0:
                raise ContractError("worker status dimension is invalid")
            if dtype != "float32":
                raise ContractError("worker status dtype must be float32")
            if normalized is not True:
                raise ContractError("worker status must declare normalized vectors")
            if not isinstance(output_sha256, str) or len(output_sha256) != 64:
                raise ContractError("worker status output_sha256 is invalid")
        return cls(
            status=status,
            job_id=request.job_id,
            request_sha256=request.sha256,
            error_type=payload.get("error_type"),
            message=payload.get("message"),
            retryable=bool(payload.get("retryable", False)),
            count=count if isinstance(count, int) else None,
            dimension=dimension if isinstance(dimension, int) else None,
            dtype=dtype if isinstance(dtype, str) else None,
            normalized=normalized if isinstance(normalized, bool) else None,
            output_sha256=output_sha256 if isinstance(output_sha256, str) else None,
        )


def status_payload_for_failure(request: WorkerRequest, error_type: str, message: str, retryable: bool) -> dict[str, object]:
    """Produce a worker status body with enough data for orchestrator audit."""

    return {
        "schema_version": "1.0.0",
        "job_id": request.job_id,
        "request_sha256": request.sha256,
        "status": "failed",
        "error_type": error_type,
        "message": message,
        "retryable": retryable,
        "attempt": request.attempt,
        "finished_at_utc": utc_now_iso8601(),
    }
