from pathlib import Path
import io
import subprocess

from tv4.clients.tv2_visual_client import TV2VisualClient
from tv4.contracts import SearchCandidate


def _candidate(query_id: str, event_index: int | None) -> SearchCandidate:
    return SearchCandidate(
        query_id=query_id, video_id="L24_V019", frame_id=13230,
        timestamp_ms=441000, source="visual", rank=1, event_index=event_index,
    )


def test_search_batch_falls_back_to_public_single_search_when_wp03_is_legacy(monkeypatch):
    client = TV2VisualClient(
        python_executable="python", wp03_cwd=Path("."), artifact_root=Path("artifacts"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stdout="", stderr="invalid choice: 'search-batch'"
        ),
    )
    calls: list[tuple[str, str, int | None]] = []

    def fake_search(query_id: str, query_text: str, event_index: int | None = None):
        calls.append((query_id, query_text, event_index))
        return [_candidate(query_id, event_index)]

    monkeypatch.setattr(client, "search", fake_search)
    requests = [("whole", "lion dance", None), ("event-1", "lands", 0)]
    result = client.search_batch(requests)

    assert calls == requests
    assert [[candidate.query_id for candidate in candidates] for candidates in result] == [["whole"], ["event-1"]]


def test_health_exposes_partial_wp03_model_availability(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    profile_path = tmp_path / "runtime.yaml"
    artifact_root = tmp_path / "artifacts"
    python_path.touch()
    profile_path.touch()
    artifact_root.mkdir()
    client = TV2VisualClient(
        python_executable=str(python_path),
        wp03_cwd=tmp_path,
        artifact_root=artifact_root,
        runtime_profile=profile_path,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=(
                '{"degraded":true,"models_requested":["beit3","metaclip2"],'
                '"models_used":["beit3"],"wp03_run_id":"run-1","candidates":[]}'
            ), stderr="",
        ),
    )

    client.search("q-1", "weighing scale")
    health = client.health()

    assert health["available"] is True
    assert health["degraded"] is True
    assert health["last_search"]["models_requested"] == ["beit3", "metaclip2"]
    assert health["last_search"]["models_used"] == ["beit3"]


def test_persistent_client_reuses_one_wp03_coordinator(monkeypatch):
    response = (
        '{"degraded":false,"models_requested":["fake"],"models_used":["fake"],'
        '"wp03_run_id":"run-1","candidates":[]}\n'
    )

    class FakeProcess:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(response + response)
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    starts = []

    def fake_popen(*args, **kwargs):
        starts.append(args[0])
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    client = TV2VisualClient(
        python_executable="python", wp03_cwd=Path("."), artifact_root=Path("artifacts"),
        persistent=True,
    )

    client.search("q-1", "first")
    client.search("q-2", "second")

    assert len(starts) == 1
    assert "serve-stdio" in starts[0]
    client.close()


def test_online_model_subset_is_forwarded_to_wp03_coordinator():
    client = TV2VisualClient(
        python_executable="python", wp03_cwd=Path("."), artifact_root=Path("artifacts"),
        persistent=True, models=("bge_vl", "metaclip2"),
    )

    command = client._server_command()

    assert command[command.index("--models") + 1] == "bge_vl,metaclip2"


def test_persistent_batch_can_override_online_models(monkeypatch):
    client = TV2VisualClient(
        python_executable="python", wp03_cwd=Path("."), artifact_root=Path("artifacts"),
        persistent=True, models=("bge_vl", "metaclip2"),
    )
    payloads = []

    def fake_request(payload):
        payloads.append(payload)
        return {"results": [{"candidates": []}]}

    monkeypatch.setattr(client, "_persistent_request", fake_request)

    client.search_batch(
        [("q", "lecture slide", None)], models=("beit3", "metaclip2"),
    )

    assert payloads[0]["models"] == "beit3,metaclip2"


def test_persistent_batch_forwards_canonical_frame_scope(monkeypatch):
    client = TV2VisualClient(
        python_executable="python", wp03_cwd=Path("."), artifact_root=Path("artifacts"),
        persistent=True,
    )
    payloads = []

    def fake_request(payload):
        payloads.append(payload)
        return {"results": [{"candidates": []}]}

    monkeypatch.setattr(client, "_persistent_request", fake_request)
    scope = [{"video_id": "L25_V062", "frame_id": 13718}]

    client.search_batch([("q", "lecture slide", None)], candidate_frames=scope)

    assert payloads[0]["candidate_frames"] == scope


def test_warmup_marks_persistent_client_ready(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    profile_path = tmp_path / "runtime.yaml"
    artifact_root = tmp_path / "artifacts"
    python_path.touch()
    profile_path.touch()
    artifact_root.mkdir()
    client = TV2VisualClient(
        python_executable=str(python_path), wp03_cwd=tmp_path,
        artifact_root=artifact_root, runtime_profile=profile_path,
        persistent=True, warmup_enabled=True,
        models=("bge_vl", "metaclip2"),
        warmup_models=("bge_vl", "metaclip2", "beit3"),
    )
    payloads = []

    def fake_request(payload, *, timeout_seconds=None):
        payloads.append((payload, timeout_seconds))
        return {
            "degraded": False,
            "models_requested": ["bge_vl", "metaclip2", "beit3"],
            "models_used": ["bge_vl", "metaclip2", "beit3"],
            "wp03_run_id": "run-ready",
            "candidates": [],
        }

    monkeypatch.setattr(client, "_persistent_request", fake_request)

    assert client.health()["available"] is False
    assert client.health()["warmup"]["status"] == "pending"
    client.start_warmup()
    client._warmup_thread.join(timeout=1)

    health = client.health()
    assert health["available"] is True
    assert health["warmup"]["status"] == "ready"
    assert payloads[0][0]["models"] == "bge_vl,metaclip2,beit3"
    assert payloads[0][1] == 180.0


def test_warmup_failure_is_visible_in_health(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    profile_path = tmp_path / "runtime.yaml"
    artifact_root = tmp_path / "artifacts"
    python_path.touch()
    profile_path.touch()
    artifact_root.mkdir()
    client = TV2VisualClient(
        python_executable=str(python_path), wp03_cwd=tmp_path,
        artifact_root=artifact_root, runtime_profile=profile_path,
        persistent=True, warmup_enabled=True,
    )

    def fail_request(payload, *, timeout_seconds=None):
        raise RuntimeError("model load timed out")

    monkeypatch.setattr(client, "_persistent_request", fail_request)
    client.start_warmup()
    client._warmup_thread.join(timeout=1)

    health = client.health()
    assert health["available"] is False
    assert health["warmup"] == {
        "enabled": True,
        "status": "failed",
        "error": "model load timed out",
    }


def test_warmup_fails_when_a_required_route_model_is_missing(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    profile_path = tmp_path / "runtime.yaml"
    artifact_root = tmp_path / "artifacts"
    python_path.touch()
    profile_path.touch()
    artifact_root.mkdir()
    client = TV2VisualClient(
        python_executable=str(python_path), wp03_cwd=tmp_path,
        artifact_root=artifact_root, runtime_profile=profile_path,
        persistent=True, warmup_enabled=True,
        models=("bge_vl", "metaclip2"),
        warmup_models=("bge_vl", "metaclip2", "beit3"),
    )
    monkeypatch.setattr(client, "_persistent_request", lambda *args, **kwargs: {
        "degraded": True,
        "models_requested": ["bge_vl", "metaclip2", "beit3"],
        "models_used": ["bge_vl", "metaclip2"],
        "candidates": [],
    })

    client.start_warmup()
    client._warmup_thread.join(timeout=1)

    health = client.health()
    assert health["available"] is False
    assert health["warmup"]["status"] == "failed"
    assert "beit3" in health["warmup"]["error"]
