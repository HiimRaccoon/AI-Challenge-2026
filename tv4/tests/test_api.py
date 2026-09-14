import os

import pytest

os.environ["TV4_FIXTURE_MODE"] = "1"

from fastapi.testclient import TestClient

from tv4.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fixture_mode(monkeypatch):
    """Keep these fixture-contract tests independent of import order.

    Other TV4 tests import ``tv4.api`` in live mode first when the suite is
    collected together.  Setting only the environment variable above is then
    too late because ``FIXTURE_MODE`` is computed at module import time.
    """
    import tv4.api as api_module

    monkeypatch.setattr(api_module, "FIXTURE_MODE", True)
    monkeypatch.setattr(api_module, "_services", None)


def test_health_fixture_mode():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "fixture"


def test_health_does_not_degrade_for_disabled_optional_dependencies(monkeypatch):
    from types import SimpleNamespace
    import tv4.api as api_module

    services = SimpleNamespace(
        preprocess_run_id="run-test",
        tv1=SimpleNamespace(health=lambda: {"status": "ok"}),
        visual=SimpleNamespace(
            health=lambda: {"available": True, "enabled": True, "degraded": False}
        ),
        tv3=SimpleNamespace(health=lambda: {"available": True, "status": "ok"}),
        dense_ocr=SimpleNamespace(
            health=lambda: {"available": False, "enabled": False, "reason": "disabled"}
        ),
        refine=None,
    )
    monkeypatch.setattr(api_module, "FIXTURE_MODE", False)
    monkeypatch.setattr(api_module, "_get_services", lambda: services)

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["degraded_reasons"] == []
    assert body["dependencies"]["dense_ocr"]["enabled"] is False
    assert body["dependencies"]["tv2_refine"]["enabled"] is False


def test_kis_search_shape():
    resp = client.post("/kis/search", json={"query_text": "một diễn giả mặc áo đỏ"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance_mode"] == "fixture"
    assert "query_id" in body and "candidates" in body
    first = body["candidates"][0]
    for field in ("video_id", "frame_id", "timestamp_ms", "rank", "score", "provenance_sources"):
        assert field in first


def test_vqa_answer_shape():
    resp = client.post("/vqa/answer", json={"query_text": "video lễ trao giải", "question": "màu ly?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance_mode"] == "fixture"
    first = body["results"][0]
    for field in (
        "video_id", "frame_id", "answer", "verified", "manual_review",
        "proposal", "approved", "verifier_status", "retry_count",
        "manual_required", "status", "evidence", "answerability_score",
        "answerability_components", "answer_confidence", "verifier_score",
        "independently_supported", "contradiction", "verification",
        "support_start_ms", "support_end_ms", "frame_candidates",
    ):
        assert field in first
    assert first["approved"] is False
    assert first["manual_required"] is True
    assert first["evidence"]["selected_frames"][0]["submission_selectable"] is False
    assert first["verification"]["visual_verifier_available"] is False
    assert first["verification"]["visual_frame_ids"] == []


def test_trake_align_shape():
    resp = client.post("/trake/align", json={"query_text": "giậm nhảy; bay qua xà; tiếp đất; đứng dậy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance_mode"] == "fixture"
    assert body["result"]["video_id"]
    assert len(body["result"]["frame_ids"]) == 4
    assert body["results"][0] == body["result"]
    assert len(body["result"]["events"]) == 4
    assert "alternatives" in body["result"]["events"][0]
    assert body["result"]["order_consistent"] is True
    assert body["result"]["verifier_score"] == 0.88
    assert "verification" in body["result"]
    assert body["result"]["verification"]["visual_verifier_available"] is False
    assert body["result"]["verification"]["visual_frame_ids"] == [[], [], [], []]


def test_request_validation_rejects_bad_strategy():
    resp = client.post("/trake/align", json={"query_text": "x", "strategy": "not-a-real-strategy"})
    assert resp.status_code == 422


def test_fixture_trake_event_rerun_preserves_locked_bounds():
    resp = client.post("/trake/rerun-event", json={
        "query_text": "event one, event two, event three",
        "events": ["event one", "event two", "event three"],
        "event_index": 1,
        "video_id": "L10_V010",
        "locked_events": [
            {"event_index": 0, "frame_id": 101, "timestamp_ms": 4040},
            {"event_index": 2, "frame_id": 203, "timestamp_ms": 8120},
        ],
        "exclude_frame_id": 156,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_index"] == 1
    assert body["lower_bound_ms"] == 4040
    assert body["upper_bound_ms"] == 8120
    assert body["alternatives"] == []


def test_live_mode_missing_config_returns_clean_500(monkeypatch):
    """Outside fixture mode, a broken/missing config must surface as a clear
    500 with a message pointing at the config path, not a raw traceback."""
    import tv4.api as api_module

    monkeypatch.setattr(api_module, "FIXTURE_MODE", False)
    monkeypatch.setattr(api_module, "CONFIG_PATH", "configs/does-not-exist.yaml")
    monkeypatch.setattr(api_module, "_services", None)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert "does-not-exist.yaml" in body["error"]

    resp = client.post("/kis/search", json={"query_text": "x"})
    assert resp.status_code == 500
    assert "does-not-exist.yaml" in resp.json()["detail"]


def test_live_kis_returns_503_while_visual_is_warming(monkeypatch):
    from types import SimpleNamespace
    import tv4.api as api_module

    visual = SimpleNamespace(health=lambda: {
        "available": False,
        "enabled": True,
        "degraded": False,
        "warmup": {"enabled": True, "status": "running", "error": None},
    })
    services = SimpleNamespace(visual=visual)
    monkeypatch.setattr(api_module, "FIXTURE_MODE", False)
    monkeypatch.setattr(api_module, "_get_services", lambda: services)

    resp = client.post("/kis/search", json={"query_text": "lecture slide"})

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "5"
    assert "running" in resp.json()["detail"]


def test_live_kis_does_not_report_visual_failure_as_empty_result(monkeypatch):
    from types import SimpleNamespace
    import tv4.api as api_module

    visual = SimpleNamespace(health=lambda: {
        "available": False,
        "enabled": True,
        "degraded": True,
        "warmup": {"enabled": False, "status": "disabled", "error": None},
        "last_search": {
            "reason": "wp03_persistent_batch_failed",
            "error": "WP03 search exceeded 34s budget",
        },
    })
    services = SimpleNamespace(visual=visual)
    monkeypatch.setattr(api_module, "FIXTURE_MODE", False)
    monkeypatch.setattr(api_module, "_get_services", lambda: services)
    monkeypatch.setattr(api_module, "run_kis_query", lambda *args, **kwargs: [])

    resp = client.post("/kis/search", json={"query_text": "lecture slide"})

    assert resp.status_code == 503
    assert "exceeded 34s budget" in resp.json()["detail"]
