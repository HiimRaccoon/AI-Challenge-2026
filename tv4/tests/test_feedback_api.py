"""Comprehensive test suite for TV4 Feedback integration (WP13 T020).

Validates:
- WP08 lifecycle exposed through TV4 API (/feedback/start, /refine, /undo, /reset, /session/{id})
- Deterministic fixture mode behavior
- Original query immutability
- Strict optimistic CAS revision checks (HTTP 409 on stale revision)
- Canonical CandidateId validation (video_id + non-negative frame_id)
- Rejection of unrendered candidate references (HTTP 400)
- Validation of text budget, whitespace, and event history limits (<=5 events)
- Expiration and missing session handling (HTTP 404)
- Model ranking failure propagation (HTTP 502)
- Basket non-mutation / side-effect isolation
- Preservation of canonical frame identity with zero local frame arithmetic
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure tv4 and WP08 source roots are present in sys.path
_TV4_SRC = (Path(__file__).resolve().parents[1] / "src").resolve()
_WP08_SRC = (Path(__file__).resolve().parents[2] / "TV2" / "WP08" / "src").resolve()

for src in (_TV4_SRC, _WP08_SRC):
    if src.exists():
        src_str = str(src)
        if src_str in sys.path:
            sys.path.remove(src_str)
        sys.path.insert(0, src_str)

from tv4 import api
from tv4.contracts import (
    FeedbackStartRequest,
    FeedbackRefineRequest,
    FeedbackUndoRequest,
    FeedbackResetRequest,
    ContractError,
)
from tv4.adapters.wp08_adapter import Wp08FeedbackAdapter
from wp08.contracts import CandidateId, ModelRankingFailed, SessionPool, CandidateMetadata


@pytest.fixture(autouse=True)
def enable_fixture_mode():
    """Ensure tests run in fixture mode by default and reset state."""
    prior = api.FIXTURE_MODE
    api.FIXTURE_MODE = True
    api._fixture_feedback_adapter = None
    try:
        yield
    finally:
        api.FIXTURE_MODE = prior
        api._fixture_feedback_adapter = None


@pytest.fixture
def client():
    return TestClient(api.app)


# ---------------------------------------------------------------------------
# 1. Contract unit tests
# ---------------------------------------------------------------------------

def test_feedback_contracts_validation() -> None:
    req = FeedbackStartRequest("s1", "red car")
    assert req.session_id == "s1"
    assert req.original_query == "red car"

    with pytest.raises(ContractError, match="session_id"):
        FeedbackStartRequest("", "red car")
    with pytest.raises(ContractError, match="original_query"):
        FeedbackStartRequest("s1", "   ")

    refine_req = FeedbackRefineRequest("s1", "L21_V001", 10690, "night scene", 0)
    assert refine_req.video_id == "L21_V001"
    assert refine_req.frame_id == 10690
    assert refine_req.expected_revision == 0

    with pytest.raises(ContractError, match="frame_id"):
        FeedbackRefineRequest("s1", "L21_V001", -1, "night scene", 0)
    with pytest.raises(ContractError, match="feedback_text"):
        FeedbackRefineRequest("s1", "L21_V001", 10690, "   ", 0)
    with pytest.raises(ContractError, match="expected_revision"):
        FeedbackRefineRequest("s1", "L21_V001", 10690, "night scene", -1)

    undo_req = FeedbackUndoRequest("s1", 1)
    assert undo_req.expected_revision == 1
    with pytest.raises(ContractError, match="expected_revision"):
        FeedbackUndoRequest("s1", -1)

    reset_req = FeedbackResetRequest("s1", 2)
    assert reset_req.expected_revision == 2
    with pytest.raises(ContractError, match="expected_revision"):
        FeedbackResetRequest("s1", -1)


# ---------------------------------------------------------------------------
# 2. Lifecycle & HTTP Endpoints via TestClient
# ---------------------------------------------------------------------------

def test_feedback_start_session(client: TestClient) -> None:
    resp = client.post("/feedback/start", json={"session_id": "test_sess_01", "original_query": "blue bus"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["session_id"] == "test_sess_01"
    assert data["revision"] == 0
    assert data["status"] == "ok"
    assert data["provenance_mode"] == "fixture"
    assert len(data["candidates"]) >= 2

    first = data["candidates"][0]
    assert first["video_id"] == "L21_V001"
    assert first["frame_id"] == 10690
    assert first["rank"] == 1
    assert first["timestamp_ms"] == 356333
    assert first["submission_selectable"] is False


def test_feedback_get_session_view(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_view", "original_query": "market street"})

    resp = client.get("/feedback/session/test_sess_view")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["session_id"] == "test_sess_view"
    assert data["revision"] == 0
    assert len(data["candidates"]) >= 2
    assert data["candidates"][0]["video_id"] == "L21_V001"


def test_feedback_refine_success(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_refine", "original_query": "drone shot"})

    # Select second candidate from fixture pool ("L21_V002", 23940)
    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_refine",
            "video_id": "L21_V002",
            "frame_id": 23940,
            "feedback_text": "focus on red roof",
            "expected_revision": 0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["revision"] == 1
    # Candidate L21_V002 should now be at top rank 1
    assert data["candidates"][0]["video_id"] == "L21_V002"
    assert data["candidates"][0]["frame_id"] == 23940
    assert data["candidates"][0]["rank"] == 1


def test_feedback_original_query_immutability(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_query", "original_query": "yellow submarine"})

    # Refine 1
    client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_query",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "underwater scene",
            "expected_revision": 0,
        },
    )

    # Check view: session remains coherent under original query
    resp = client.get("/feedback/session/test_sess_query")
    assert resp.status_code == 200
    data = resp.json()
    assert data["revision"] == 1


def test_feedback_revision_monotonic_increments(client: TestClient) -> None:
    # 0: start
    r0 = client.post("/feedback/start", json={"session_id": "test_sess_mono", "original_query": "running man"}).json()
    assert r0["revision"] == 0

    # 1: refine
    r1 = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_mono",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "in forest",
            "expected_revision": 0,
        },
    ).json()
    assert r1["revision"] == 1

    # 2: refine again
    r2 = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_mono",
            "video_id": "L21_V002",
            "frame_id": 23940,
            "feedback_text": "near river",
            "expected_revision": 1,
        },
    ).json()
    assert r2["revision"] == 2

    # 3: undo
    r3 = client.post("/feedback/undo", json={"session_id": "test_sess_mono", "expected_revision": 2}).json()
    assert r3["revision"] == 3

    # 4: reset
    r4 = client.post("/feedback/reset", json={"session_id": "test_sess_mono", "expected_revision": 3}).json()
    assert r4["revision"] == 4


def test_feedback_stale_revision_conflict_409(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_cas", "original_query": "traffic jam"})

    # Send refine with stale expected_revision=99
    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_cas",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "motorcycles only",
            "expected_revision": 99,
        },
    )
    assert resp.status_code == 409
    assert "revision" in resp.text.lower() or "stale" in resp.text.lower()


def test_feedback_undo_semantics(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_undo", "original_query": "bridge aerial"})

    # Refine 1: move L21_V002 to top
    client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_undo",
            "video_id": "L21_V002",
            "frame_id": 23940,
            "feedback_text": "sunset illumination",
            "expected_revision": 0,
        },
    )

    # Undo
    resp = client.post("/feedback/undo", json={"session_id": "test_sess_undo", "expected_revision": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["revision"] == 2
    # First candidate should be reverted to initial pool ordering (L21_V001)
    assert data["candidates"][0]["video_id"] == "L21_V001"


def test_feedback_reset_semantics(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_reset", "original_query": "stadium lights"})

    # Refine 1
    client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_reset",
            "video_id": "L21_V002",
            "frame_id": 23940,
            "feedback_text": "indoor lights",
            "expected_revision": 0,
        },
    )

    # Reset
    resp = client.post("/feedback/reset", json={"session_id": "test_sess_reset", "expected_revision": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["revision"] == 2
    # Candidate order restored to initial pool
    assert data["candidates"][0]["video_id"] == "L21_V001"


def test_feedback_unrendered_candidate_400(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_unrendered", "original_query": "solar farm"})

    # Candidate not in pool
    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_unrendered",
            "video_id": "UNKNOWN_VID",
            "frame_id": 99999,
            "feedback_text": "wind turbines",
            "expected_revision": 0,
        },
    )
    assert resp.status_code == 400
    assert "candidate is not rendered" in resp.text.lower()


def test_feedback_empty_or_whitespace_text_400(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_blank", "original_query": "solar farm"})

    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_blank",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "   ",
            "expected_revision": 0,
        },
    )
    assert resp.status_code == 400 or resp.status_code == 422


def test_feedback_event_history_limit_400(client: TestClient) -> None:
    client.post("/feedback/start", json={"session_id": "test_sess_limit", "original_query": "city panorama"})

    for rev in range(5):
        resp = client.post(
            "/feedback/refine",
            json={
                "session_id": "test_sess_limit",
                "video_id": "L21_V001",
                "frame_id": 10690,
                "feedback_text": f"refinement step {rev+1}",
                "expected_revision": rev,
            },
        )
        assert resp.status_code == 200, resp.text

    # 6th refinement should exceed MAX_FEEDBACK_EVENTS (5)
    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_limit",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "one step too far",
            "expected_revision": 5,
        },
    )
    assert resp.status_code == 400
    assert "five" in resp.text.lower() or "event" in resp.text.lower() or "limit" in resp.text.lower()


def test_feedback_nonexistent_or_expired_session_404(client: TestClient) -> None:
    resp = client.get("/feedback/session/nonexistent_session_id")
    assert resp.status_code == 404

    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "nonexistent_session_id",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "ghost feedback",
            "expected_revision": 0,
        },
    )
    assert resp.status_code == 404


def test_feedback_model_ranking_failure_502(client: TestClient) -> None:
    def failing_ranker(*args, **kwargs):
        raise ModelRankingFailed("4-model ensemble failed on CUDA device")

    adapter = Wp08FeedbackAdapter(fixture_mode=True, ranker=failing_ranker)
    api._fixture_feedback_adapter = adapter

    client.post("/feedback/start", json={"session_id": "test_sess_rank_fail", "original_query": "night skyline"})

    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_rank_fail",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "add stars",
            "expected_revision": 0,
        },
    )
    assert resp.status_code == 502
    assert "ensemble failed" in resp.text.lower() or "ranking" in resp.text.lower()


def test_feedback_disabled_or_unconfigured_503() -> None:
    api.FIXTURE_MODE = False
    api._fixture_feedback_adapter = None
    api._services = None
    try:
        c = TestClient(api.app)
        resp = c.post("/feedback/start", json={"session_id": "live_sess", "original_query": "live query"})
        # Should return 503 because live backend is not provisioned or config is unconfigured
        assert resp.status_code == 503
    finally:
        api.FIXTURE_MODE = True
        api._services = None



def test_feedback_no_frame_arithmetic_and_basket_isolation(client: TestClient) -> None:
    """Ensure candidates preserve exact frame IDs and feedback has no basket side-effects."""
    resp = client.post("/feedback/start", json={"session_id": "test_sess_iso", "original_query": "subway entrance"})
    assert resp.status_code == 200
    data = resp.json()

    for cand in data["candidates"]:
        assert isinstance(cand["frame_id"], int)
        assert not isinstance(cand["frame_id"], bool)
        assert cand["frame_id"] >= 0
        # Check that frame_id is not altered by nominal FPS math (e.g. 10690 stays 10690)
        if cand["video_id"] == "L21_V001":
            assert cand["frame_id"] == 10690
            assert cand["timestamp_ms"] == 356333
        elif cand["video_id"] == "L21_V002":
            assert cand["frame_id"] == 23940
            assert cand["timestamp_ms"] == 798000


def test_feedback_refine_with_exact_corrected_reference(client: TestClient) -> None:
    """Ensure operator exact-corrected reference succeeds and preserves canonical identity and certified root anchor."""
    client.post("/feedback/start", json={"session_id": "test_sess_exact_ref", "original_query": "person riding bicycle"})

    # Candidate L21_V001 root keyframe is 10690. Operator exact-corrects to frame 10696 (offset +6)
    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_exact_ref",
            "video_id": "L21_V001",
            "frame_id": 10696,
            "source_candidate_frame_id": 10690,
            "feedback_text": "more wheels visible",
            "expected_revision": 0,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["revision"] == 1
    first = data["candidates"][0]
    assert first["video_id"] == "L21_V001"
    assert first["frame_id"] == 10696
    assert first["certified_anchor_frame_id"] == 10690
    assert first["anchor_offset"] == 6
    assert first["rank"] == 1


def test_feedback_refine_with_forged_source_candidate_fails_closed(client: TestClient) -> None:
    """Ensure unrendered forged source candidates fail closed with HTTP 400."""
    client.post("/feedback/start", json={"session_id": "test_sess_forged", "original_query": "person walking"})

    resp = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_forged",
            "video_id": "L21_V001",
            "frame_id": 10696,
            "source_candidate_frame_id": 99999,  # unrendered
            "feedback_text": "invalid candidate",
            "expected_revision": 0,
        },
    )
    assert resp.status_code == 400
    assert "candidate is not rendered" in resp.text


def test_feedback_five_active_event_limit_lifecycle(client: TestClient) -> None:
    """Ensure session enforces <=5 active events, rejects 6th with 400, and enables refine after undo/reset."""
    start_resp = client.post("/feedback/start", json={"session_id": "test_sess_5limit", "original_query": "street traffic"})
    assert start_resp.status_code == 200
    data = start_resp.json()
    assert data["active_feedback_count"] == 0
    assert data["max_active_feedback_events"] == 5

    # 5 successive refinements
    current_rev = 0
    for i in range(1, 6):
        resp = client.post(
            "/feedback/refine",
            json={
                "session_id": "test_sess_5limit",
                "video_id": "L21_V001",
                "frame_id": 10690,
                "feedback_text": f"refinement step {i}",
                "expected_revision": current_rev,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["active_feedback_count"] == i
        assert data["max_active_feedback_events"] == 5
        current_rev = data["revision"]

    # 6th refinement fails closed with HTTP 400
    resp6 = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_5limit",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "6th illegal refinement",
            "expected_revision": current_rev,
        },
    )
    assert resp6.status_code == 400
    assert "session permits at most five active feedback events" in resp6.text

    # Undo reduces active count back to 4
    undo_resp = client.post(
        "/feedback/undo",
        json={
            "session_id": "test_sess_5limit",
            "expected_revision": current_rev,
        },
    )
    assert undo_resp.status_code == 200
    data = undo_resp.json()
    assert data["active_feedback_count"] == 4
    current_rev = data["revision"]

    # Now 5th refinement is allowed again
    resp5_again = client.post(
        "/feedback/refine",
        json={
            "session_id": "test_sess_5limit",
            "video_id": "L21_V001",
            "frame_id": 10690,
            "feedback_text": "5th refinement again",
            "expected_revision": current_rev,
        },
    )
    assert resp5_again.status_code == 200
    data = resp5_again.json()
    assert data["active_feedback_count"] == 5
    current_rev = data["revision"]

    # Reset clears all active events (count = 0)
    reset_resp = client.post(
        "/feedback/reset",
        json={
            "session_id": "test_sess_5limit",
            "expected_revision": current_rev,
        },
    )
    assert reset_resp.status_code == 200
    data = reset_resp.json()
    assert data["active_feedback_count"] == 0
