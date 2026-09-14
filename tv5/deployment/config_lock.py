"""WP13 configuration lock with SHA-256 digest and secret exclusion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from tv5.telemetry.events import redact_secrets


@dataclass(frozen=True)
class ConfigLock:
    config_id: str
    digest_sha256: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sanitized_config: dict[str, Any] = field(default_factory=dict)
    service_urls: dict[str, str] = field(default_factory=dict)
    read_only_mounts: list[str] = field(default_factory=list)
    mutable_volumes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_config_lock(
    config_id: str,
    raw_config: dict[str, Any],
    service_urls: dict[str, str],
    read_only_mounts: list[str] | None = None,
    mutable_volumes: list[str] | None = None,
) -> ConfigLock:
    """Generate deterministic config lock with secret exclusion and SHA-256 digest."""
    sanitized = redact_secrets(raw_config)
    serialized = json.dumps(sanitized, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return ConfigLock(
        config_id=config_id,
        digest_sha256=digest,
        sanitized_config=sanitized,
        service_urls=dict(service_urls),
        read_only_mounts=list(read_only_mounts or []),
        mutable_volumes=list(mutable_volumes or []),
    )


def verify_config_lock(lock: ConfigLock, current_config: dict[str, Any]) -> bool:
    """Verify that current configuration matches the locked digest."""
    sanitized = redact_secrets(current_config)
    serialized = json.dumps(sanitized, sort_keys=True)
    current_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return current_digest == lock.digest_sha256
