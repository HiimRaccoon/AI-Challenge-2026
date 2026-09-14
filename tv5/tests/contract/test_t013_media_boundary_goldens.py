"""T013 characterization guard for the existing WP04 media boundary."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys

GOLDEN = Path(__file__).with_name("t013_media_boundary_goldens.json")


def test_t013_media_characterization_golden_exists_and_is_machine_readable() -> None:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["task"] == "T013"
    assert data["classification"] == "CODE GAP"


def _sources() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[3]
    return (
        repo.parent / "tv1tv3" / "TV1_TV3_WP04",
        repo / "tv4" / "src" / "tv4" / "api.py",
    )


def _media_row(video_id: str, path: Path) -> dict[str, object]:
    return {
        "preprocess_run_id": "t013-run",
        "video_id": video_id,
        "original_video_path": str(path),
        "source_sha256": "0" * 64,
        "duration_ms": 1,
        "width_px": 1,
        "height_px": 1,
        "has_audio": False,
        "created_at_utc": "2026-08-18T00:00:00Z",
    }


def _wp04_client(tmp_path: Path):
    wp04, _ = _sources()
    sys.path.insert(0, str(wp04 / "src"))
    from aic2026.api import create_app
    from fastapi.testclient import TestClient

    media_root = tmp_path / "synthetic-media-root"
    media_root.mkdir()
    mp4 = media_root / "fixture.mp4"
    mp4.write_bytes(b"0123456789")
    txt = media_root / "not-media.txt"
    txt.write_bytes(b"text-media-is-not-rejected")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside-root")
    link = media_root / "escape-link.mp4"
    try:
        os.symlink(outside, link)
    except OSError:
        link = None
    run_root = tmp_path / "runs" / "t013-run"
    (run_root / "media").mkdir(parents=True)
    rows = [_media_row("VID1", mp4), _media_row("TXT1", txt), _media_row("ESCAPE1", outside)]
    if link is not None:
        rows.append(_media_row("LINK1", link))
    (run_root / "media" / "media.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (run_root / "frames.jsonl").write_text("", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n  runs_root: " + str(tmp_path / "runs").replace("\\", "/")
        + "\nevidence_catalog:\n  enabled: false\n",
        encoding="utf-8",
    )
    return TestClient(create_app("t013-run", config)), mp4, outside, link


def test_wp04_actual_http_range_and_read_only_behavior(tmp_path: Path) -> None:
    client, source, _, _ = _wp04_client(tmp_path)
    before = (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns)
    with client:
        head = client.head("/videos/VID1/stream")
        full = client.get("/videos/VID1/stream")
        partial = client.get("/videos/VID1/stream", headers={"Range": "bytes=2-5"})
        suffix = client.get("/videos/VID1/stream", headers={"Range": "bytes=-3"})
        invalid = [client.get("/videos/VID1/stream", headers={"Range": value}) for value in ("bytes=99-", "bytes=5-2", "bytes=0-1,3-4", "items=0-1", "bytes=--")]
    assert head.status_code == 200
    assert head.headers["accept-ranges"] == "bytes"
    assert head.headers["content-length"] == "10"
    assert full.status_code == 200 and full.content == b"0123456789"
    assert partial.status_code == 206 and partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert suffix.status_code == 206 and suffix.content == b"789"
    assert all(response.status_code == 416 for response in invalid)
    assert all(response.headers["content-range"] == "bytes */10" for response in invalid)
    assert before == (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns)


def test_wp04_current_registry_boundary_and_unenforced_policies(tmp_path: Path) -> None:
    client, _, outside, link = _wp04_client(tmp_path)
    with client:
        unknown = client.get("/videos/UNKNOWN/stream")
        traversal = client.get("/videos/%2E%2E%2Foutside/stream")
        disallowed_extension = client.get("/videos/TXT1/stream")
        absolute_escape = client.get("/videos/ESCAPE1/stream")
        symlink_escape = client.get("/videos/LINK1/stream") if link is not None else None
    assert unknown.status_code == 404
    assert traversal.status_code == 404
    # The upstream service resolves only registered video_id values, but trusts
    # each record's absolute path: it has no configured media root or extension allow-list.
    assert disallowed_extension.status_code == 200
    assert disallowed_extension.content == b"text-media-is-not-rejected"
    assert absolute_escape.status_code == 200 and absolute_escape.content == outside.read_bytes()

    if link is None:
        # Windows denied symlink creation; source inspection below still proves
        # no resolved-path containment check exists in this WP04 route.
        source = (_sources()[0] / "src" / "aic2026" / "api.py").read_text(encoding="utf-8")
        stream_body = source[source.index("def _video_path"):source.index("@app.api_route")]
        assert ".resolve(" not in stream_body and "relative_to(" not in stream_body
    else:
        assert symlink_escape is not None
        assert symlink_escape.status_code == 200 and symlink_escape.content == outside.read_bytes()


def test_wp04_frame_identity_and_timestamp_semantics_are_source_backed() -> None:
    wp04, tv4_api = _sources()
    api = (wp04 / "src" / "aic2026" / "api.py").read_text(encoding="utf-8")
    media = (wp04 / "src" / "aic2026" / "media.py").read_text(encoding="utf-8")
    tv4 = tv4_api.read_text(encoding="utf-8")
    assert '@app.get("/videos/{video_id}/frames/{frame_id}.jpg")' in api
    for header in ("X-Original-Frame-Id", "X-PTS", "X-Timestamp-Ms"):
        assert header in api
    assert '@app.get("/videos/{video_id}/resolve")' in api
    assert 'mode not in {"nearest", "before", "after"}' in api
    assert "resolve_timestamp_record(timestamp_ms, mode=mode)" in api
    assert "Resolve and decode frames on the original PTS timeline" in media
    assert "def get_frame_with_record" in media
    assert '"/videos/{video_id}/stream"' in api
    # T013 is a historical characterization: its golden retains the then-open
    # TV4 transport gap, while later T014 may safely close that gap.
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    historical_gap = {item["capability"]: item["tv4_exposure"] for item in golden["capabilities"]}
    assert historical_gap["Original-video streaming"] == "No /videos media route in tracked TV4 api.py."
    assert historical_gap["Original-frame JPEG and timestamp/frame resolution"] == "No browser frame-image or resolve transport route in tracked TV4."
    assert '"/exact-frame/neighbors"' in tv4
