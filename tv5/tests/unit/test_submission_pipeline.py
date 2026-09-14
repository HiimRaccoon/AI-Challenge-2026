"""Tests for submission basket, CSV export/validation, ZIP packaging, and CLI fallback."""
from __future__ import annotations

from pathlib import Path
import pytest
import zipfile

from tv5.submission import (
    Basket,
    KisPrediction,
    VqaPrediction,
    TrakePrediction,
    export_kis_csv,
    export_vqa_csv,
    export_trake_csv,
    parse_submission_csv,
    validate_csv_file,
    validate_prediction,
    validate_submission_package,
    package_submission_zip,
)
from tv5.submission.cli import main as cli_main


def test_kis_basket_and_export_roundtrip() -> None:
    basket = Basket(query_id="kis-001", task_type="KIS")
    for i in range(100):
        basket.add(KisPrediction(video_id=f"L01_V{i:03d}", frame_id=i * 25, rank=i + 1))

    # Reject 101st
    assert not basket.add(KisPrediction(video_id="L01_V999", frame_id=999, rank=101))
    assert len(basket.items) == 100

    audit = basket.audit()
    assert audit.is_valid, audit.errors

    csv_text = export_kis_csv([item for item in basket.items if isinstance(item, KisPrediction)])
    rows = parse_submission_csv(csv_text)
    assert len(rows) == 100
    assert rows[0] == ["L01_V000", "0"]
    assert rows[99] == ["L01_V099", "2475"]


def test_vqa_basket_approval_guard_and_escaping() -> None:
    # Unapproved VQA fails validation
    unapproved = VqaPrediction(video_id="L02_V001", frame_id=120, approved_answer="cốc màu đỏ", approved=False)
    assert not validate_prediction(unapproved).is_valid

    # Approved VQA with commas, quotes, Vietnamese text
    complex_answer = 'Cốc màu đỏ, có chữ "AIC 2026"\nvà hoa văn'
    approved = VqaPrediction(
        video_id="L02_V001",
        frame_id=120,
        approved_answer=complex_answer,
        approved=True,
    )
    assert validate_prediction(approved).is_valid

    csv_text = export_vqa_csv([approved])
    rows = parse_submission_csv(csv_text)
    assert len(rows) == 1
    assert rows[0][0] == "L02_V001"
    assert rows[0][1] == "120"
    assert rows[0][2] == complex_answer  # exact text preserved on roundtrip


def test_trake_basket_order_and_event_count_guard() -> None:
    # Less than 2 events fails
    bad_trake = TrakePrediction(video_id="L03_V001", event_frame_ids=(100,), expected_event_count=4)
    assert not validate_prediction(bad_trake).is_valid

    # Exactly 4 events preserved in semantic order
    valid_trake = TrakePrediction(
        video_id="L03_V001",
        event_frame_ids=(100, 250, 400, 550),
        expected_event_count=4,
    )
    assert validate_prediction(valid_trake).is_valid

    csv_text = export_trake_csv([valid_trake])
    rows = parse_submission_csv(csv_text)
    assert len(rows) == 1
    assert rows[0] == ["L03_V001", "100", "250", "400", "550"]


def test_csv_validator_rejects_malformed_and_headers(tmp_path: Path) -> None:
    # Header rejection
    p_header = tmp_path / "header.csv"
    p_header.write_text("video_id,frame_id\nL01_V001,100\n", encoding="utf-8")
    assert not validate_csv_file(p_header, task_type="KIS").is_valid

    # .mp4 rejection
    p_mp4 = tmp_path / "bad_ext.csv"
    p_mp4.write_text("L01_V001.mp4,100\n", encoding="utf-8")
    assert not validate_csv_file(p_mp4, task_type="KIS").is_valid

    # Negative frame_id
    p_neg = tmp_path / "neg.csv"
    p_neg.write_text("L01_V001,-5\n", encoding="utf-8")
    assert not validate_csv_file(p_neg, task_type="KIS").is_valid


def test_submission_packager_and_reopen_validation(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()

    # Create 3 valid query CSVs
    (csv_dir / "query_1.csv").write_text("L01_V001,100\nL01_V002,200\n", encoding="utf-8")
    (csv_dir / "query_2.csv").write_text('L01_V001,100,"màu đỏ"\n', encoding="utf-8")
    (csv_dir / "query_3.csv").write_text("L01_V001,100,200,300\n", encoding="utf-8")

    zip_path = tmp_path / "submission.zip"
    report = package_submission_zip(
        {"query_1": csv_dir / "query_1.csv", "query_2": csv_dir / "query_2.csv", "query_3": csv_dir / "query_3.csv"},
        zip_path,
    )
    assert report.is_valid, report.errors
    assert report.record_count == 3
    assert report.package_digest_sha256 is not None

    # Inspect zip structure
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert all(n.startswith("submission/") for n in names)
        assert "submission/query_1.csv" in names
        assert "submission/query_2.csv" in names
        assert "submission/query_3.csv" in names


def test_package_validator_rejects_root_csv_and_bad_zip(tmp_path: Path) -> None:
    bad_zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip_path, "w") as zf:
        zf.writestr("query_root.csv", "L01_V001,100\n")

    report = validate_submission_package(bad_zip_path)
    assert not report.is_valid
    assert any("outside required top-level 'submission/'" in e for e in report.errors)


def test_cli_fallback_commands(tmp_path: Path) -> None:
    csv_path = tmp_path / "q1.csv"
    csv_path.write_text("L01_V001,100\n", encoding="utf-8")

    # CLI validate-csv
    rc = cli_main(["validate-csv", str(csv_path), "--task-type", "KIS"])
    assert rc == 0

    # CLI check-api
    rc_api = cli_main(["check-api"])
    assert rc_api == 0
