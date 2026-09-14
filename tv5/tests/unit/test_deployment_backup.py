"""Tests for deployment configuration locking and state backup/restore recovery."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tv5.deployment import generate_config_lock, verify_config_lock
from tv5.backup import create_backup, restore_backup


def test_config_lock_digest_and_secret_redaction() -> None:
    raw_cfg = {
        "model_version": "v1.2",
        "api_key": "secret-12345",
        "max_candidates": 100,
        "nested": {"db_password": "supersecretpassword", "timeout_s": 30},
    }
    service_urls = {"tv1": "http://127.0.0.1:8000", "wp04": "http://127.0.0.1:8100"}
    lock = generate_config_lock("cfg-001", raw_cfg, service_urls)

    assert lock.config_id == "cfg-001"
    assert lock.sanitized_config["api_key"] == "[REDACTED]"
    assert lock.sanitized_config["nested"]["db_password"] == "[REDACTED]"
    assert lock.digest_sha256 is not None

    # Verification matches
    assert verify_config_lock(lock, raw_cfg) is True

    # Tampered config fails verification
    tampered = dict(raw_cfg)
    tampered["max_candidates"] = 500
    assert verify_config_lock(lock, tampered) is False


def test_backup_and_restore_roundtrip_with_manifest_and_exclusions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Valid mutable state files
    (state_dir / "draft_query.json").write_text('{"query": "test"}', encoding="utf-8")
    (state_dir / "basket.json").write_text('{"items": [1, 2, 3]}', encoding="utf-8")

    # Files that MUST be excluded (large upstream assets)
    (state_dir / "test_video.mp4").write_bytes(b"mock mp4 data")
    (state_dir / "index.faiss").write_bytes(b"mock faiss index")

    backup_zip = tmp_path / "backup.zip"
    manifest = create_backup(state_dir, backup_zip, backup_id="test_backup_1")

    assert manifest.file_count == 2
    assert "draft_query.json" in manifest.file_checksums
    assert "basket.json" in manifest.file_checksums
    assert "test_video.mp4" not in manifest.file_checksums
    assert "index.faiss" not in manifest.file_checksums
    assert manifest.readiness_invalidated is True

    # Restore to a fresh directory
    restored_dir = tmp_path / "restored_state"
    ok, res_manifest, errors = restore_backup(backup_zip, restored_dir)

    assert ok is True
    assert not errors
    assert res_manifest is not None
    assert (restored_dir / "draft_query.json").read_text(encoding="utf-8") == '{"query": "test"}'
    assert (restored_dir / "basket.json").read_text(encoding="utf-8") == '{"items": [1, 2, 3]}'
    assert not (restored_dir / "test_video.mp4").exists()
    assert not (restored_dir / "index.faiss").exists()
    assert res_manifest.readiness_invalidated is True
