from __future__ import annotations
import json
from pathlib import Path
import sys

GOLDEN = Path(__file__).with_name("current_boundary_goldens.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tv4" / "src"))

def test_current_supported_boundaries_are_explicit() -> None:
    data = json.loads(GOLDEN.read_text())
    assert data["supported"]["feedback"]["status"] == "SUPPORTED"
    assert data["supported"]["exact_neighbors"]["refuse"]
    assert "frame_id" in data["supported"]["kis"]["canonical"]

def test_tv4_source_matches_golden_endpoint_set() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (root / "tv4" / "src" / "tv4" / "api.py").read_text(encoding="utf-8")
    data = json.loads(GOLDEN.read_text())
    for name in ("health", "kis", "vqa", "trake", "exact_neighbors", "feedback"):
        assert f'"{data["supported"][name]["path"]}"' in source
    assert '"/feedback/refine"' in source
    assert '"/feedback/undo"' in source
    assert '"/feedback/reset"' in source

def test_actual_tv4_contract_serialization_and_exact_rejection() -> None:
    from tv4.contracts import SearchCandidate, exact_neighbor_response_is_safe
    candidate = SearchCandidate.from_json({"query_id":"q","video_id":"v","frame_id":1,"timestamp_ms":2,"source":"visual","rank":1,"unknown_future_field":True})
    assert candidate.to_json()["frame_id"] == 1
    assert not exact_neighbor_response_is_safe({"provenance_mode":"fixture"}, "v", 1, [0], "run")

def test_actual_request_models_and_fixture_health() -> None:
    from tv4 import api
    from tv4.api import KisRequest, VqaRequest, TrakeRequest
    assert KisRequest(query_text="x").top_k == 100
    assert VqaRequest(query_text="x", question="q").top_k_answers == 5
    assert TrakeRequest(query_text="x").strategy == "dp"
    prior = api.FIXTURE_MODE; api.FIXTURE_MODE = True
    try: assert api.health() == {"status":"ok", "mode":"fixture"}
    finally: api.FIXTURE_MODE = prior

def test_actual_wp08_contract_exists_but_tv4_feedback_remains_gap() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "TV2" / "WP08" / "src"))
    from wp08.contracts import CandidateId, FeedbackEvent, FeedbackValidationError, SessionView
    event = FeedbackEvent.create(candidate_id=CandidateId("V", 1), feedback_text="at night")
    assert event.candidate_id.frame_id == 1
    assert SessionView("s", 0, ()).revision == 0
    try: FeedbackEvent.create(candidate_id=CandidateId("V", 1), feedback_text=" ")
    except FeedbackValidationError: pass
    else: raise AssertionError("WP08 must reject blank feedback")
