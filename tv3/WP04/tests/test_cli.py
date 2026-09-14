import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wp04.cli import _runtime_batch_size, build_preprocess_fingerprint, main
from wp04.contracts import AudioRecord, FrameRecord, OCRDetection, WP04RunIdentity
from wp04.storage import ArtifactStore


def test_cli_search_emits_tv4_candidate_json(tmp_path: Path, capsys):
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))
    ocr = OCRDetection("tv1", "v", 42, 1400, "BÁNH MÌ", "bánh mì", (0.1, 0.2, 0.3, 0.4), 0.9, "m", "v", "ocr:v:42:0")
    store.commit_video("ocr", "v", [ocr], "fp")
    assert main([
        "search", "--run-dir", str(tmp_path), "--preprocess-run-id", "tv1", "--artifact-set-id", "wp04",
        "--source", "ocr", "--query", "banh mi",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"][0]["source"] == "ocr"


def test_cli_doctor_accepts_direct_http_transport_without_commands(capsys, monkeypatch):
    for variable in ("WP04_OCR_COMMAND", "WP04_CHUNKFORMER_COMMAND", "WP04_RFDETR_COMMAND"):
        monkeypatch.delenv(variable, raising=False)
    config = Path(__file__).parents[1] / "configs" / "default.yaml"
    assert main(["doctor", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["production_ready"] is True
    assert report["checks"]["ocr"]["transport"] == "http"
    assert report["checks"]["ocr"]["status"] == "ok"


def test_cli_doctor_accepts_ocr_only_config(capsys, monkeypatch):
    monkeypatch.setenv("WP04_OCR_COMMAND", "ocr-client --keyframe-path {keyframe_path}")
    for variable in ("WP04_CHUNKFORMER_COMMAND", "WP04_RFDETR_COMMAND"):
        monkeypatch.delenv(variable, raising=False)
    config = Path(__file__).parents[1] / "configs" / "ocr-only.yaml"
    assert main(["doctor", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["asr"]["status"] == "disabled"
    assert report["checks"]["object"]["status"] == "disabled"


def test_cli_exports_legacy_tv1_parquet_manifests(tmp_path: Path, capsys):
    run_dir = tmp_path / "tv1" / "data" / "runs" / "run-a"
    image = run_dir / "keyframes" / "v" / "0.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"keyframe-bytes")
    audio_file = run_dir / "media" / "v.wav"
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"audio-bytes")
    pq.write_table(pa.Table.from_pylist([{
        "preprocess_run_id": "run-a", "video_id": "v", "frame_id": 0,
        "keyframe_seq": 0, "timestamp_ms": 0, "keyframe_path": "keyframes/v/0.jpg",
    }]), run_dir / "frames.parquet")
    pq.write_table(pa.Table.from_pylist([{
        "preprocess_run_id": "run-a", "video_id": "v", "audio_path": "data/runs/run-a/media/v.wav",
        "audio_sha256": "audio-sha", "duration_ms": 1000, "status": "ready",
    }]), run_dir / "media" / "audio.parquet")

    output_dir = tmp_path / "wp04-inputs"
    assert main(["export-tv1-manifests", "--tv1-run-dir", str(run_dir), "--output-dir", str(output_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    frames, audio = json.loads((output_dir / "frames.json").read_text()), json.loads((output_dir / "audio.json").read_text())
    assert report["frame_count"] == 1
    assert frames[0]["keyframe_path"] == str(image.resolve())
    assert len(frames[0]["keyframe_sha256"]) == 64
    assert frames[0]["shot_id"] is None
    assert frames[0]["selection_reason"] is None
    assert audio[0]["audio_path"] == str(audio_file.resolve())


def test_cli_validate_writes_and_emits_promotion_gate(tmp_path: Path, capsys):
    frames = tmp_path / "frames.json"
    frames.write_text(json.dumps([{
        "preprocess_run_id": "tv1", "video_id": "v", "frame_id": 42, "keyframe_seq": 1, "timestamp_ms": 1400,
        "keyframe_path": "keyframes/v/000042.jpg", "keyframe_sha256": "frame-sha",
    }]), encoding="utf-8")
    assert main([
        "validate", "--run-dir", str(tmp_path), "--preprocess-run-id", "tv1", "--artifact-set-id", "wp04",
        "--frames-json", str(frames),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_cli_preprocess_surfaces_unconfigured_models_but_keeps_no_audio_valid(tmp_path: Path, capsys):
    frames = tmp_path / "frames.json"
    frames.write_text(json.dumps([{
        "preprocess_run_id": "tv1", "video_id": "v", "frame_id": 42, "keyframe_seq": 1, "timestamp_ms": 1400,
        "keyframe_path": "keyframes/v/000042.jpg", "keyframe_sha256": "frame-sha",
    }]), encoding="utf-8")
    audio = tmp_path / "audio.json"
    audio.write_text(json.dumps([{
        "preprocess_run_id": "tv1", "video_id": "v", "audio_path": None, "checksum": None,
        "declared_present": False, "duration_ms": None,
    }]), encoding="utf-8")
    config = Path(__file__).parents[1] / "configs" / "default.yaml"
    assert main([
        "preprocess", "--run-dir", str(tmp_path), "--preprocess-run-id", "tv1", "--artifact-set-id", "wp04",
        "--frames-json", str(frames), "--audio-json", str(audio), "--config", str(config),
    ]) == 0
    states = {item["modality"]: item["state"] for item in json.loads(capsys.readouterr().out)["statuses"]}
    assert states == {"ocr": "failed", "asr": "no_audio", "object": "failed", "metadata": "ready"}


def test_preprocess_fingerprint_changes_with_keyframe_bytes():
    audio = AudioRecord("tv1", "v", None, None, False, None)
    first = FrameRecord("tv1", "v", 42, 1, 1400, "keyframes/42.jpg", "first")
    second = FrameRecord("tv1", "v", 42, 1, 1400, "keyframes/42.jpg", "second")
    config = {"normalization_version": "v", "modalities": {}}
    assert build_preprocess_fingerprint([first], {"v": audio}, config) != build_preprocess_fingerprint([second], {"v": audio}, config)


def test_runtime_batch_size_override_is_positive(monkeypatch):
    monkeypatch.setenv("WP04_RUNTIME_OCR_BATCH_SIZE", "16")
    assert _runtime_batch_size("WP04_RUNTIME_OCR_BATCH_SIZE", 8) == 16
    monkeypatch.setenv("WP04_RUNTIME_OCR_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="positive"):
        _runtime_batch_size("WP04_RUNTIME_OCR_BATCH_SIZE", 8)
