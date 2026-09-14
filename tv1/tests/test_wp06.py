"""
Tests for WP06 — Validation Engine & API Server
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from dataclasses import asdict
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wp06_api_server import (
    ValidationIssue,
    PreprocessingRun,
    validate_keyframe_mapping,
    validate_frame_records,
    run_full_validation,
    create_preprocessing_run,
    save_preprocessing_run,
    load_preprocessing_run,
    update_run_status,
    app,
    GLOBAL_RUN_DIR,
)
import wp06_api_server


# ── Validation Tests ──

class TestValidationIssue:
    def test_create_issue(self):
        issue = ValidationIssue(
            severity="P0",
            module="wp02",
            video_id="L21_V001",
            message="Missing keyframe file",
        )
        assert issue.severity == "P0"
        assert issue.module == "wp02"
        assert issue.video_id == "L21_V001"


class TestKeyframeValidation:
    def test_missing_frames_parquet(self, tmp_path):
        """P0 if frames.parquet doesn't exist."""
        issues = validate_keyframe_mapping(tmp_path)
        assert any(i.severity == "P0" for i in issues)

    def test_valid_frames(self, tmp_path):
        """No issues if everything is valid."""
        # Create frames.parquet with valid data
        (tmp_path / "keyframes" / "test_vid").mkdir(parents=True)
        img_path = tmp_path / "keyframes" / "test_vid" / "0.jpg"
        # Create a minimal valid JPEG
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(str(img_path), "JPEG")
        
        df = pd.DataFrame([{
            "schema_version": "1.0.0",
            "preprocess_run_id": "test_run",
            "video_id": "test_vid",
            "frame_id": 0,
            "keyframe_seq": 0,
            "timestamp_ms": 0,
            "pts": 0,
            "shot_id": "test_vid_shot_000",
            "keyframe_path": "keyframes/test_vid/0.jpg",
            "thumbnail_path": None,
            "selection_reason": "shot_representative",
            "sharpness_score": 100.0,
            "blur_score": 0.01,
            "created_at_utc": "2026-01-01T00:00:00Z",
        }])
        df.to_parquet(tmp_path / "frames.parquet")
        
        issues = validate_keyframe_mapping(tmp_path)
        p0_issues = [i for i in issues if i.severity == "P0"]
        assert len(p0_issues) == 0, f"Unexpected P0 issues: {p0_issues}"


# ── Registry Tests ──

class TestPreprocessingRegistry:
    def test_create_and_save(self, tmp_path):
        run = create_preprocessing_run(
            preprocess_run_id="test_run",
            source_manifest_sha256="abc123",
            config_sha256="def456",
            code_commit="abc123def",
            artifact_root=str(tmp_path),
        )
        assert run.preprocess_run_id == "test_run"
        assert run.status == "running"
        
        save_preprocessing_run(run, tmp_path)
        assert (tmp_path / "manifest.json").exists()

    def test_load_roundtrip(self, tmp_path):
        run = create_preprocessing_run(
            preprocess_run_id="roundtrip_test",
            source_manifest_sha256="abc",
            config_sha256="def",
            code_commit="123",
            artifact_root=str(tmp_path),
        )
        save_preprocessing_run(run, tmp_path)
        loaded = load_preprocessing_run(tmp_path)
        assert loaded.preprocess_run_id == "roundtrip_test"
        assert loaded.status == "running"

    def test_update_status(self, tmp_path):
        run = create_preprocessing_run(
            preprocess_run_id="status_test",
            source_manifest_sha256="abc",
            config_sha256="def",
            code_commit="123",
            artifact_root=str(tmp_path),
        )
        save_preprocessing_run(run, tmp_path)
        update_run_status(tmp_path, "partial")
        loaded = load_preprocessing_run(tmp_path)
        assert loaded.status == "partial"

    def test_stable_run_guard(self, tmp_path):
        run = create_preprocessing_run(
            preprocess_run_id="guard_test",
            source_manifest_sha256="abc",
            config_sha256="def",
            code_commit="123",
            artifact_root=str(tmp_path),
        )
        run.status = "stable"
        save_preprocessing_run(run, tmp_path)
        
        with pytest.raises((ValueError, Exception)):
            update_run_status(tmp_path, "running")


# ── API Tests ──

class TestAPI:
    @pytest.fixture
    def client(self, tmp_path):
        """Create a test client using the module-level app and GLOBAL_RUN_DIR."""
        from fastapi.testclient import TestClient
        (tmp_path / "manifest").mkdir(parents=True)
        (tmp_path / "media").mkdir(parents=True)
        (tmp_path / "logs").mkdir(parents=True)
        # Set the global run dir to the tmp_path
        wp06_api_server.GLOBAL_RUN_DIR = tmp_path
        with TestClient(app) as c:
            yield c

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_summary(self, client):
        resp = client.get("/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_videos" in data
        assert "total_frames" in data

    def test_manifest_404_when_empty(self, client):
        resp = client.get("/manifest")
        assert resp.status_code == 404

    def test_cors(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
