"""Stable fingerprints for resumable, auditable WP04 artifacts."""

from __future__ import annotations

from hashlib import sha256
from json import dumps
from typing import Any, Mapping


def build_input_fingerprint(
    input_checksums: Mapping[str, Any], effective_config: Mapping[str, Any], adapter_versions: Mapping[str, str],
    *, normalization_version: str = "vi-unicode-lower-fold-v1",
) -> str:
    payload = {
        "input_checksums": input_checksums,
        "effective_config": effective_config,
        "adapter_versions": adapter_versions,
        "normalization_version": normalization_version,
    }
    encoded = dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(encoded).hexdigest()
