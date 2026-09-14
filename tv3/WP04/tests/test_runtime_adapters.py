import subprocess
import json

import pytest

from wp04.adapters import AdapterUnavailableError
from wp04.contracts import FrameRecord
from wp04.runtime_adapters import build_ocr_adapter


def test_command_ocr_adapter_reports_missing_local_command_before_inference():
    adapter = build_ocr_adapter(command_env="MISSING_WP04_OCR_COMMAND")
    frame = FrameRecord("tv1", "v", 42, 1, 1400, "keyframes/42.jpg", "sha")
    with pytest.raises(AdapterUnavailableError, match="MISSING_WP04_OCR_COMMAND"):
        adapter.detect(frame)


def test_command_adapter_reads_utf8_json_from_local_model(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0, '{"detections": []}', "")

    monkeypatch.setenv("WP04_TEST_OCR_COMMAND", "local-model --path {keyframe_path}")
    monkeypatch.setattr("wp04.runtime_adapters.subprocess.run", fake_run)
    adapter = build_ocr_adapter(command_env="WP04_TEST_OCR_COMMAND")
    frame = FrameRecord("tv1", "v", 42, 1, 1400, "keyframes/42.jpg", "sha")

    assert adapter.detect(frame) == []
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_ocr_adapter_posts_batches_over_one_persistent_model_server_connection(monkeypatch):
    captured = {"requests": []}

    class FakeResponse:
        status = 200

        def read(self):
            return b'{"detections": [[{"text": "banh mi", "bbox_xyxy_norm": [0, 0, 1, 1], "confidence": 0.9}], []]}'

    class FakeConnection:
        def __init__(self, host, port, timeout):
            captured.setdefault("connections", []).append((host, port, timeout))

        def request(self, method, path, body, headers):
            captured["requests"].append((method, path, json.loads(body), headers))

        def getresponse(self):
            return FakeResponse()

        def close(self):
            captured["closed"] = captured.get("closed", 0) + 1

    monkeypatch.setattr("wp04.runtime_adapters.http.client.HTTPConnection", FakeConnection)
    adapter = build_ocr_adapter(command_env="MISSING", server_url="http://127.0.0.1:8765")
    frames = [
        FrameRecord("tv1", "v", 1, 1, 100, "keyframes/1.jpg", "sha1"),
        FrameRecord("tv1", "v", 2, 2, 200, "keyframes/2.jpg", "sha2"),
    ]

    rows = adapter.detect_batch(frames)
    adapter.detect_batch(frames)

    assert captured["connections"] == [("127.0.0.1", 8765, 120)]
    assert [request[:3] for request in captured["requests"]] == [
        ("POST", "/ocr/batch", {"keyframe_paths": ["keyframes/1.jpg", "keyframes/2.jpg"]}),
        ("POST", "/ocr/batch", {"keyframe_paths": ["keyframes/1.jpg", "keyframes/2.jpg"]}),
    ]
    assert captured["requests"][0][3]["Connection"] == "keep-alive"
    assert rows[0][0].text == "banh mi"
    assert rows[1] == []
