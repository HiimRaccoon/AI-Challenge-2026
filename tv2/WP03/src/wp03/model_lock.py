"""Content locks for model checkpoints that cannot be identified by revision alone."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import sha256_file, write_json_atomically
from .contracts import ContractError


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    source_url: str
    revision: str = ""


@dataclass(frozen=True)
class ModelLock:
    path: Path
    filename: str
    source_url: str
    expected_size_bytes: int
    sha256: str
    identity: ModelIdentity


def create_model_lock(
    checkpoint_path: Path,
    lock_path: Path,
    identity: ModelIdentity,
    expected_size_bytes: int,
) -> ModelLock:
    """Lock a team-verified checkpoint only after enforcing its official size."""

    if not checkpoint_path.is_file():
        raise ContractError("checkpoint does not exist")
    actual_size = checkpoint_path.stat().st_size
    if actual_size != expected_size_bytes:
        raise ContractError(f"checkpoint size mismatch: expected {expected_size_bytes}, got {actual_size}")
    digest = sha256_file(checkpoint_path)
    lock = ModelLock(
        path=lock_path,
        filename=checkpoint_path.name,
        source_url=identity.source_url,
        expected_size_bytes=expected_size_bytes,
        sha256=digest,
        identity=identity,
    )
    write_json_atomically(
        lock_path,
        {
            "schema_version": "1.0.0",
            "model_name": identity.name,
            "revision": identity.revision,
            "source_url": identity.source_url,
            "filename": lock.filename,
            "expected_size_bytes": expected_size_bytes,
            "sha256": digest,
        },
    )
    return lock


def validate_model_lock(lock_path: Path, checkpoint_path: Path) -> ModelLock:
    """Reject absent, stale, changed-size, or changed-content checkpoints."""

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("model lock is absent or unreadable") from exc
    if not checkpoint_path.is_file():
        raise ContractError("checkpoint does not exist")
    expected_size = payload.get("expected_size_bytes")
    if not isinstance(expected_size, int) or checkpoint_path.stat().st_size != expected_size:
        raise ContractError("checkpoint size does not match model lock")
    digest = sha256_file(checkpoint_path)
    if digest != payload.get("sha256"):
        raise ContractError("checkpoint digest does not match model lock")
    return ModelLock(
        path=lock_path,
        filename=str(payload.get("filename", "")),
        source_url=str(payload.get("source_url", "")),
        expected_size_bytes=expected_size,
        sha256=digest,
        identity=ModelIdentity(
            name=str(payload.get("model_name", "")),
            source_url=str(payload.get("source_url", "")),
            revision=str(payload.get("revision", "")),
        ),
    )
