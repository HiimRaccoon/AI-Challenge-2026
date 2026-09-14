"""T014 browser-media boundary tests (synthetic files only)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi.testclient import TestClient

from tv4 import api
from tv4.kis_pipeline import KisServices
from tv4.media_identity import MediaRecord


class _Noop:
    pass


class _TV1Frames:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def frames(self, video_id: str) -> list[dict]:
        return [record for record in self.records if record.get("video_id") == video_id]


def _client(monkeypatch, tmp_path: Path) -> tuple[TestClient, Path]:
    root = tmp_path / "raw"
    root.mkdir()
    source = root / "VID1.mp4"
    source.write_bytes(b"0123456789")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    text = root / "text.txt"
    text.write_text("not video", encoding="utf-8")
    escape = root / "escape.mp4"
    try:
        os.symlink(outside, escape)
    except OSError:
        pass
    registry = {
        "VID1": MediaRecord("VID1", "VID1.mp4", "a" * 64, "run", "1/1000", "m/VID1", "f/VID1"),
        "OUT": MediaRecord("OUT", str(outside), "a" * 64, "run", "1/1000", "m/OUT", "f/OUT"),
        "TXT": MediaRecord("TXT", "text.txt", "a" * 64, "run", "1/1000", "m/TXT", "f/TXT"),
        "ESC": MediaRecord("ESC", "escape.mp4", "a" * 64, "run", "1/1000", "m/ESC", "f/ESC"),
    }
    services = KisServices(
        tv1=_TV1Frames([]), tv3=_Noop(), visual=_Noop(), preprocess_run_id="run",
        original_media_root=root, media_registry=registry,
        allowed_media_extensions=frozenset({".mp4"}),
    )
    monkeypatch.setattr(api, "FIXTURE_MODE", False)
    monkeypatch.setattr(api, "_get_services", lambda: services)
    return TestClient(api.app), source


def test_t014_stream_is_registry_bound_range_safe_and_read_only(monkeypatch, tmp_path: Path) -> None:
    client, source = _client(monkeypatch, tmp_path)
    before = (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns)
    with client:
        head = client.head("/videos/VID1/stream")
        full = client.get("/videos/VID1/stream")
        closed = client.get("/videos/VID1/stream", headers={"Range": "bytes=2-5"})
        opened = client.get("/videos/VID1/stream", headers={"Range": "bytes=7-"})
        suffix = client.get("/videos/VID1/stream", headers={"Range": "bytes=-3"})
        invalid = [client.get("/videos/VID1/stream", headers={"Range": value}) for value in ("bytes=99-", "bytes=5-2", "bytes=0-1,3-4", "items=0-1", "bytes=--")]
    assert head.status_code == 200 and head.headers["accept-ranges"] == "bytes" and head.headers["content-length"] == "10" and head.content == b""
    assert full.status_code == 200 and full.content == b"0123456789"
    assert closed.status_code == 206 and closed.content == b"2345" and closed.headers["content-range"] == "bytes 2-5/10"
    assert opened.status_code == 206 and opened.content == b"789"
    assert suffix.status_code == 206 and suffix.content == b"789"
    assert all(item.status_code == 416 and item.headers["content-range"] == "bytes */10" for item in invalid)
    assert before == (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns)


def test_t014_rejects_untrusted_registry_paths_extensions_and_client_paths(monkeypatch, tmp_path: Path) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    with client:
        assert client.get("/videos/UNKNOWN/stream").status_code == 404
        assert client.get("/videos/%2E%2E%2Foutside/stream").status_code == 404
        assert client.get("/videos/OUT/stream").status_code == 404
        assert client.get("/videos/TXT/stream").status_code == 404
        assert client.get("/videos/ESC/stream").status_code == 404
        assert client.get("/videos/VID1/stream", params={"path": str(tmp_path / "outside.mp4")}).status_code == 200


def test_t014_selected_images_and_timestamp_fail_closed_without_wp04(monkeypatch, tmp_path: Path) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    assets = tmp_path / "assets"; assets.mkdir()
    key = assets / "key.jpg"; thumb = assets / "thumb.jpg"
    key.write_bytes(b"key"); thumb.write_bytes(b"thumb")
    services = api._get_services()
    services.derivative_asset_root = assets
    services.allowed_image_extensions = frozenset({".jpg"})
    services.tv1 = _TV1Frames([{
        "video_id": "VID1", "frame_id": 7, "timestamp_ms": 71, "pts": 70,
        "preprocess_run_id": "run", "keyframe_path": str(key), "thumbnail_path": str(thumb),
    }])
    with client:
        selected = client.get("/videos/VID1/keyframes/7.jpg")
        thumbnail = client.get("/videos/VID1/thumbnails/7.jpg")
        original = client.get("/videos/VID1/frames/7.jpg")
        unresolved = client.get("/videos/VID1/resolve", params={"timestamp_ms": 71, "mode": "nearest"})
    assert selected.status_code == 200 and selected.content == b"key"
    assert thumbnail.status_code == 200 and thumbnail.content == b"thumb"
    assert original.status_code == 200 and original.content == b"key"
    assert original.headers["x-original-frame-id"] == "7"
    assert unresolved.status_code == 200
    assert unresolved.json() == {
        "video_id": "VID1", "requested_timestamp_ms": 71, "mode": "nearest",
        "canonical_frame_id": None, "canonical_timestamp_ms": None,
        "inspection_only": True, "submission_selectable": False,
        "state": "canonical_frame_unresolved_from_arbitrary_playback_timestamp",
    }
    assert not hasattr(api, "_upstream_media_get")
    assert not hasattr(services, "media_api_base_url")


def test_t014_exact_neighbors_remain_wp04_off_and_wp09_proven(monkeypatch, tmp_path: Path) -> None:
    client, _ = _client(monkeypatch, tmp_path)

    class Authority:
        certification_id = "e4-1"
        certification_report_sha256 = "c" * 64

    class Refiner:
        exact_proof_authority = Authority()
        def __init__(self): self.calls = 0
        def neighbors(self, request):
            self.calls += 1
            offset = request["offsets"][0]
            return {"provenance_mode": "live", "video_id": "VID1", "anchor_frame_id": 1, "steps": [{"offset": offset, "frame": {
                "video_id": "VID1", "frame_id": 1 + offset, "timestamp_ms": 1 + offset, "pts": 100 + offset,
                "time_base": "1/1000", "preprocess_run_id": "run", "mapping_guaranteed": True,
                "submission_selectable": True, "identity_source": "certified_run_consecutive_original_decode",
                "media_identity_verified": True, "producer_compatibility_verified": True,
                "certification_id": "e4-1", "certification_report_sha256": "c" * 64,
                "source_sha256": "a" * 64,
            }}]}

    refiner = Refiner()
    services = api._get_services(); services.refine = refiner
    with client:
        compatible = client.post("/exact-frame/neighbors", json={"video_id": "VID1", "frame_id": 1, "timestamp_ms": 1, "offsets": [1]})
        disallowed = client.post("/exact-frame/neighbors", json={"video_id": "TXT", "frame_id": 1, "timestamp_ms": 1, "offsets": [0]})
    assert refiner.calls == 1
    assert compatible.json()["steps"][0]["frame"]["frame_id"] == 2
    assert disallowed.json()["degraded_reason"] == "original media extension is not allowed"


def test_t014_serves_only_contained_tv1_selected_derivatives(monkeypatch, tmp_path: Path) -> None:
    client, _ = _client(monkeypatch, tmp_path)
    assets = tmp_path / "assets"; assets.mkdir()
    key = assets / "key.jpg"; thumb = assets / "thumb.jpg"
    key.write_bytes(b"key"); thumb.write_bytes(b"thumb")
    services = api._get_services(); services.derivative_asset_root = assets
    services.allowed_image_extensions = frozenset({".jpg"})
    frame = {"video_id": "VID1", "frame_id": 7, "timestamp_ms": 71, "preprocess_run_id": "run", "keyframe_path": str(key), "thumbnail_path": str(thumb)}
    services.tv1 = _TV1Frames([frame])
    with client:
        assert client.get("/videos/VID1/keyframes/7.jpg").content == b"key"
        assert client.get("/videos/VID1/thumbnails/7.jpg").content == b"thumb"
    escaped = tmp_path / "outside.jpg"; escaped.write_bytes(b"outside")
    bad_frame = dict(frame); bad_frame["keyframe_path"] = str(escaped)
    services.tv1 = _TV1Frames([bad_frame])
    assert client.get("/videos/VID1/keyframes/7.jpg").status_code == 404
    disallowed = assets / "thumb.txt"; disallowed.write_text("no", encoding="utf-8")
    bad_frame = dict(frame); bad_frame["thumbnail_path"] = str(disallowed)
    services.tv1 = _TV1Frames([bad_frame])
    assert client.get("/videos/VID1/thumbnails/7.jpg").status_code == 404
