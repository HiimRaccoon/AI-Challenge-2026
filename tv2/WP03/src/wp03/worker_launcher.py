"""Subprocess-backed encoder used by the normal CLI, not CPU test fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import ContractError
from .worker_protocol import WorkerRequest, WorkerStatus


@dataclass
class WorkerProcessEncoder:
    command: tuple[str, ...]
    job_root: Path
    model_key: str
    revision: str
    device: str
    dtype: str
    batch_size: int
    timeout_seconds: int = 900
    fallback_dtype: str | None = None
    expected_dimension: int | None = None
    _compatibility_fingerprint: str = ""
    environment: dict[str, str] | None = None

    def _run(self, operation: str, *, image_paths: Sequence[Path] = (), texts: Sequence[str] = ()) -> np.ndarray:
        self.job_root.mkdir(parents=True, exist_ok=True)
        batch_size = self.batch_size
        dtype = self.dtype
        # A 12 GB GPU can have substantially different safe batch sizes per
        # model. Retry by progressively halving the batch instead of giving up
        # after a single reduction.
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            request = WorkerRequest.create(
                self.job_root, operation, self.model_key, self.revision, self.device, dtype,
                batch_size, attempt, image_paths, texts,
            )
            request.write()
            try:
                worker_env = {**os.environ, **(self.environment or {})}
                source_root = worker_env.pop("WP03_SOURCE_ROOT", None)
                if source_root:
                    inherited = worker_env.get("PYTHONPATH", "")
                    worker_env["PYTHONPATH"] = os.pathsep.join(
                        value for value in (source_root, inherited) if value
                    )
                completed = subprocess.run(
                    [*self.command, str(request.request_path)], capture_output=True, text=True,
                    timeout=self.timeout_seconds, check=False,
                    env=worker_env,
                )
            except subprocess.TimeoutExpired as exc:
                raise ContractError(f"worker {self.model_key} timed out") from exc
            if not request.status_path.is_file():
                raise ContractError(f"worker {self.model_key} exited without status ({completed.returncode})")
            status = WorkerStatus.from_json(request.status_path, request)
            if completed.returncode == 0 and status.status == "ok":
                if not request.output_path.is_file():
                    raise ContractError("successful worker output is absent")
                actual_hash = hashlib.sha256(request.output_path.read_bytes()).hexdigest()
                if actual_hash != status.output_sha256:
                    raise ContractError("worker output digest does not match status")
                with request.output_path.open("rb") as stream:
                    result = np.asarray(np.load(stream, allow_pickle=False), dtype=np.float32)
                if result.ndim != 2 or result.shape != (status.count, status.dimension):
                    raise ContractError("worker output shape does not match status")
                if self.expected_dimension is not None and result.shape[1] != self.expected_dimension:
                    raise ContractError("worker output dimension does not match configuration")
                return result
            if status.retryable and status.error_type == "unsupported_bfloat16" and self.fallback_dtype and dtype != self.fallback_dtype:
                dtype = self.fallback_dtype
                continue
            if status.retryable and status.error_type == "cuda_oom" and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                continue
            raise ContractError(f"worker {self.model_key} failed: {status.error_type}: {status.message}")
        raise ContractError(f"worker {self.model_key} exhausted retry policy")

    def compatibility_fingerprint(self) -> str:
        return self._compatibility_fingerprint

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        return self._run("encode_images", image_paths=image_paths)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return self._run("encode_text", texts=texts)
