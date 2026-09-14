"""Integration tests for deterministic fixture coverage and current-artifact verification."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tv5.readiness import ReadinessConfig, validate_readiness

WORKSPACE_ROOT = Path("D:/aic226")


def test_t049_fixture_mode_contract_determinism() -> None:
    from tv4 import fixtures
    assert fixtures.KIS_RESPONSE["provenance_mode"] == "fixture"
    assert len(fixtures.KIS_RESPONSE["candidates"]) > 0
    assert fixtures.EXACT_NEIGHBOR_RESPONSE["provenance_mode"] == "fixture"
    assert fixtures.TRAKE_RESPONSE["provenance_mode"] == "fixture"


def test_t050_current_wp03_physical_artifact_readiness() -> None:
    wp03_full_root = WORKSPACE_ROOT / "tv2_1" / "WP03" / "artifacts" / "full-run-1"
    if not wp03_full_root.exists():
        pytest.skip(f"WP03 full run artifact root not found at {wp03_full_root}")

    manifest_file = wp03_full_root / "manifests" / "metaclip2.json"
    if not manifest_file.exists():
        pytest.skip(f"Manifest not found: {manifest_file}")

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data.get("preprocess_run_id") == "run_v1_batch1"
    assert manifest_data.get("vector_count") == 106380 or manifest_data.get("vectors") == 106380


def test_t050_current_wp09_certification_readiness() -> None:
    cert_path = WORKSPACE_ROOT / "tv2_1" / "WP09" / "configs" / "certifications" / "run_v1_batch1.json"
    if not cert_path.exists():
        cert_path = WORKSPACE_ROOT / "tv5" / "TV2" / "WP09" / "configs" / "certifications" / "run_v1_batch1.json"

    assert cert_path.exists()
    cert_data = json.loads(cert_path.read_text(encoding="utf-8"))
    assert cert_data.get("preprocess_run_id") == "run_v1_batch1"
    assert cert_data.get("decision") in ("CERTIFIED", "PROVISIONAL_CERTIFIED") or cert_data.get("status") in ("CERTIFIED", "PROVISIONAL_CERTIFIED")


def test_t050_wp04_handover_pending_classification() -> None:
    wp04_runs_root = WORKSPACE_ROOT / "tv1tv3" / "TV1_TV3_WP04" / "data" / "runs"
    # Modality artifacts are currently pending external handover
    cfg = ReadinessConfig(
        component="OCR",
        kind="wp04",
        root=wp04_runs_root / "run_v1_batch1" / "ocr",
        known_handover=True,
        modality="OCR",
    )
    report = validate_readiness(cfg)
    assert report.status == "HANDOVER PENDING"
