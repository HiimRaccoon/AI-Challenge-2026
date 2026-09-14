"""
Tests for WP00 — Data Intake & Corpus Manifest
"""
from __future__ import annotations

import json
import hashlib
import zipfile
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wp00_data_intake import DataIntake


# ── Helpers ──

def _create_dummy_mp4(path: Path, content: bytes = b"fake_mp4_data_12345") -> None:
    """Create a minimal dummy .mp4 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _create_test_zip(zip_path: Path, files: dict[str, bytes]) -> None:
    """Create a ZIP archive with given files."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_config(tmp_path, archives_dir="archives", raw_dir="raw", run_dir="run"):
    """Build a config dict compatible with the current DataIntake API."""
    return {
        'corpus': {
            'archives_dir': str(tmp_path / archives_dir),
            'raw_dir': str(tmp_path / raw_dir),
            'video_glob': '**/*.mp4',
            'video_id_rule': 'stem',
        },
        'preprocessing': {
            'run_dir': str(tmp_path / run_dir),
        }
    }


# ── Test SHA-256 ──

class TestSHA256:
    def test_sha256_deterministic(self, tmp_path):
        f = tmp_path / "test.mp4"
        f.write_bytes(b"hello world")
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake._prepare_directories()
        intake.logger = __import__('logging').getLogger("test")
        h1 = intake._compute_sha256(f)
        h2 = intake._compute_sha256(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_matches_hashlib(self, tmp_path):
        content = b"test content for sha256"
        f = tmp_path / "test.mp4"
        f.write_bytes(content)
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake._prepare_directories()
        intake.logger = __import__('logging').getLogger("test")
        result = intake._compute_sha256(f)
        expected = _sha256(content)
        assert result == expected


# ── Test ZIP Extraction ──

class TestExtraction:
    def test_extract_zip_safe(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "test.zip", {
            "video1.mp4": b"video1_data",
            "subdir/video2.mp4": b"video2_data",
        })
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake._prepare_directories()
        intake.logger = __import__('logging').getLogger("test")
        intake._extract_zips()
        raw_dir = Path(config['corpus']['raw_dir'])
        mp4s = list(raw_dir.rglob("*.mp4"))
        assert len(mp4s) >= 1

    def test_zipslip_blocked(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "evil.zip", {
            "../../../evil.mp4": b"evil_data",
        })
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake._prepare_directories()
        intake.logger = __import__('logging').getLogger("test")
        intake._extract_zips()
        evil_path = tmp_path / "evil.mp4"
        assert not evil_path.exists()

    def test_skip_junk_files(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "test.zip", {
            "video.mp4": b"video_data",
            "__MACOSX/video.mp4": b"junk",
            ".DS_Store": b"junk",
            "readme.txt": b"text",
        })
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake._prepare_directories()
        intake.logger = __import__('logging').getLogger("test")
        intake._extract_zips()
        raw_dir = Path(config['corpus']['raw_dir'])
        all_files = list(raw_dir.rglob("*"))
        names = [f.name for f in all_files if f.is_file()]
        assert ".DS_Store" not in names
        assert "readme.txt" not in names

    def test_idempotent_extraction(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "test.zip", {"video.mp4": b"data"})
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake._prepare_directories()
        intake.logger = __import__('logging').getLogger("test")
        intake._extract_zips()
        raw_dir = Path(config['corpus']['raw_dir'])
        files1 = list(raw_dir.rglob("*.mp4"))
        intake._extract_zips()
        files2 = list(raw_dir.rglob("*.mp4"))
        assert len(files1) == len(files2)


# ── Test Duplicate Detection ──

class TestDuplicateDetection:
    def test_detect_duplicates(self, tmp_path):
        same_content = b"identical_video_content"
        _create_test_zip(tmp_path / "archives" / "a.zip", {"v1.mp4": same_content})
        _create_test_zip(tmp_path / "archives" / "b.zip", {"v2.mp4": same_content})
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake.run()
        # One should be accepted, the other duplicate
        assert len(intake.duplicates) >= 1

    def test_no_duplicates_different_content(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "a.zip", {"v1.mp4": b"content_A"})
        _create_test_zip(tmp_path / "archives" / "b.zip", {"v2.mp4": b"content_B"})
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake.run()
        assert len(intake.duplicates) == 0


# ── Test Output Files ──

class TestOutputFiles:
    def test_all_four_outputs(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "test.zip", {
            "v1.mp4": b"video_1",
            "v2.mp4": b"video_2",
        })
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake.run()

        run_dir = Path(config['preprocessing']['run_dir'])
        manifest_dir = run_dir / "manifest"
        assert (manifest_dir / "corpus_manifest.parquet").exists()
        assert (manifest_dir / "corpus_manifest.json").exists()
        assert (run_dir / "duplicate_videos.jsonl").exists()
        assert (run_dir / "rejected_files.jsonl").exists()


# ── Test Schema ──

class TestSchema:
    def test_manifest_schema_fields(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "test.zip", {"video.mp4": b"data"})
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake.run()

        import pandas as pd
        run_dir = Path(config['preprocessing']['run_dir'])
        df = pd.read_parquet(run_dir / "manifest" / "corpus_manifest.parquet")

        required_fields = [
            "schema_version", "video_id", "source_archive",
            "original_video_path", "source_sha256", "file_size_bytes",
            "batch_id", "duplicate_of_video_id", "ingest_status",
            "created_at_utc"
        ]
        for field in required_fields:
            assert field in df.columns, f"Missing field: {field}"

    def test_ingest_status_values(self, tmp_path):
        same = b"same_content"
        _create_test_zip(tmp_path / "archives" / "a.zip", {"v1.mp4": same})
        _create_test_zip(tmp_path / "archives" / "b.zip", {"v2.mp4": same})
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake.run()

        import pandas as pd
        run_dir = Path(config['preprocessing']['run_dir'])
        df = pd.read_parquet(run_dir / "manifest" / "corpus_manifest.parquet")

        statuses = set(df["ingest_status"].values)
        assert statuses.issubset({"accepted", "duplicate", "rejected"})
        assert "accepted" in statuses

    def test_video_id_no_extension(self, tmp_path):
        _create_test_zip(tmp_path / "archives" / "test.zip", {"my_video.mp4": b"data"})
        config = _make_config(tmp_path)
        intake = DataIntake(config)
        intake.run()

        import pandas as pd
        run_dir = Path(config['preprocessing']['run_dir'])
        df = pd.read_parquet(run_dir / "manifest" / "corpus_manifest.parquet")

        for vid in df["video_id"]:
            assert not vid.endswith(".mp4"), f"video_id should not have .mp4: {vid}"
