"""T019 Feedback seam characterization and contract invariant tests.

Authority:
- AIC2026_Pipeline_Final.md §2.2, §3
- WP13 specs/001-contest-ready-wp13/contracts/wp13-boundaries.md
- WP13 specs/001-contest-ready-wp13/spec.md FR-059, FR-060, User Story 13
- TV2/WP08 tracked source contracts.py, service.py, store.py, text.py, ranking.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
import pytest

# Ensure TV4 and WP08 tracked sources are importable
TV5_ROOT = Path(__file__).resolve().parents[3]
TV4_SRC = TV5_ROOT / "tv4" / "src"
WP08_SRC = TV5_ROOT / "TV2" / "WP08" / "src"

if str(TV4_SRC) not in sys.path:
    sys.path.insert(0, str(TV4_SRC))
if str(WP08_SRC) not in sys.path:
    sys.path.insert(0, str(WP08_SRC))

from wp08.contracts import (
    CandidateId,
    CandidateMetadata,
    FeedbackEvent,
    FeedbackValidationError,
    ModelRankingFailed,
    RevisionConflict,
    SessionExpired,
    SessionPool,
    SessionView,
)
from wp08.service import FeedbackSessions
from wp08.store import SqliteSessionStore
from wp08.text import build_feedback_template, validate_token_budget

GOLDEN = Path(__file__).with_name("t019_feedback_boundary_goldens.json")


def _fixed_clock(offset_seconds: int = 0):
    base = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)
    return base + timedelta(seconds=offset_seconds)


def _make_fixture_pool(candidates: list[tuple[str, int]] | None = None) -> SessionPool:
    cands = candidates or [("L21_V001", 100), ("L21_V001", 200), ("L21_V002", 300), ("L21_V003", 400)]
    cid_list = tuple(CandidateId(vid, fid) for vid, fid in cands)
    meta_list = tuple(
        CandidateMetadata(cid, idx * 1000, f"thumbnails/{cid.video_id}_{cid.frame_id:06d}.jpg")
        for idx, cid in enumerate(cid_list, start=1)
    )
    return SessionPool(
        wp03_run_id="wp03-smoke-run-001",
        candidates=cid_list,
        candidate_metadata=meta_list,
        snapshot={"fixture": True, "pool_size": len(cid_list)},
        provenance={"model": "fixture-model", "rrf_k": 60},
    )


def _create_test_service(
    tmp_path: Path,
    clock_fn=None,
    pool_provider=None,
    ranker=None,
    token_counter=None,
    renderer=None,
) -> FeedbackSessions:
    clock = clock_fn or (lambda: _fixed_clock(0))
    provider = pool_provider or (lambda query: _make_fixture_pool())
    # Default ranker: moves selected candidate to top, reverses the rest
    def default_ranker(candidates, template, selected, snapshot):
        cands = list(candidates)
        if selected and selected in cands:
            cands.remove(selected)
            return (selected, *reversed(cands))
        return tuple(reversed(cands))

    t_counter = token_counter or (lambda text: len(text.split()))

    return FeedbackSessions(
        store=SqliteSessionStore(tmp_path / "sessions.db", clock=clock),
        clock=clock,
        pool_provider=provider,
        ranker=ranker or default_ranker,
        token_counter=t_counter,
        renderer=renderer,
    )


# ---------------------------------------------------------------------------
# 1. Golden file validation
# ---------------------------------------------------------------------------


def test_t019_feedback_golden_contract_is_valid_and_machine_readable() -> None:
    assert GOLDEN.exists(), "t019_feedback_boundary_goldens.json must exist"
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["task"] == "T019"
    assert data["classification"]["wp08_core_capability"] == "SUPPORTED"
    assert data["classification"]["tv4_feedback_seam"] == "CODE GAP"
    assert data["classification"]["basket_isolation"] == "SUPPORTED"
    assert "start_session" in data["contracts"]["wp08_request_models"]
    assert "refine" in data["contracts"]["wp08_request_models"]
    assert "undo" in data["contracts"]["wp08_request_models"]
    assert "reset" in data["contracts"]["wp08_request_models"]
    assert "SessionView" in data["contracts"]["wp08_response_models"]


# ---------------------------------------------------------------------------
# 2. Basic Feedback Lifecycle & Rendered Candidate Representation
# ---------------------------------------------------------------------------


def test_wp08_basic_feedback_lifecycle(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)
    session_id = "session-basic-001"
    original_query = "red sports car at night"

    # Start session -> revision 0
    view0 = service.start_session(session_id, original_query)
    assert isinstance(view0, SessionView)
    assert view0.session_id == session_id
    assert view0.revision == 0
    assert len(view0.candidates) == 4
    assert view0.wp03_run_id == "wp03-smoke-run-001"
    assert view0.candidates[0].candidate_id == CandidateId("L21_V001", 100)
    assert view0.candidates[0].display_rank == 1
    assert view0.candidates[0].timestamp_ms == 1000
    assert view0.candidates[0].keyframe_path == "thumbnails/L21_V001_000100.jpg"

    # Refine with selected frame -> revision 1
    selected = CandidateId("L21_V002", 300)
    view1 = service.refine(session_id, selected, "front view near traffic light", expected_revision=0)
    assert isinstance(view1, SessionView)
    assert view1.revision == 1
    assert len(view1.candidates) == 4
    # Selected candidate moved to top rank
    assert view1.candidates[0].candidate_id == selected
    assert view1.candidates[0].display_rank == 1
    assert view1.candidates[0].timestamp_ms == 3000

    # View / get_session returns current state
    current = service.view(session_id)
    assert current.revision == 1
    assert current.candidates[0].candidate_id == selected


# ---------------------------------------------------------------------------
# 3. Original Query Immutability & Template Construction
# ---------------------------------------------------------------------------


def test_wp08_original_query_immutability_and_template_construction(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)
    session_id = "session-query-immutability"
    original_query = "pedestrian with blue umbrella"

    service.start_session(session_id, original_query)

    # Refinement 1
    sel1 = CandidateId("L21_V001", 200)
    service.refine(session_id, sel1, "closer zoom", expected_revision=0)

    # Refinement 2
    sel2 = CandidateId("L21_V003", 400)
    service.refine(session_id, sel2, "daylight rain", expected_revision=1)

    # Verify template construction invariant
    events = (
        FeedbackEvent.create(candidate_id=sel1, feedback_text="closer zoom"),
        FeedbackEvent.create(candidate_id=sel2, feedback_text="daylight rain"),
    )
    template = build_feedback_template(original_query, events)
    expected_template = (
        "Original query: pedestrian with blue umbrella\n"
        "Refinement 1: closer zoom\n"
        "Refinement 2: daylight rain"
    )
    assert template == expected_template

    # Inspect persisted record: original_query must remain unchanged
    _, _, _, state = service._store.get_record(session_id)
    assert state["original_query"] == original_query
    assert len(state["events"]) == 2
    assert state["events"][0]["text"] == "closer zoom"
    assert state["events"][1]["text"] == "daylight rain"


# ---------------------------------------------------------------------------
# 4. Canonical Reference Identity Preservation & Rejection
# ---------------------------------------------------------------------------


def test_wp08_canonical_reference_identity_and_rejection_of_noncanonical(tmp_path: Path) -> None:
    # Valid candidate identity
    valid_cid = CandidateId("L21_V001", 100)
    assert valid_cid.video_id == "L21_V001"
    assert valid_cid.frame_id == 100

    # Non-canonical identity: empty video_id fails closed
    with pytest.raises(FeedbackValidationError, match="candidate identity is invalid"):
        CandidateId("", 100)

    with pytest.raises(FeedbackValidationError, match="candidate identity is invalid"):
        CandidateId("   ", 100)

    # Non-canonical identity: negative frame_id fails closed
    with pytest.raises(FeedbackValidationError, match="candidate identity is invalid"):
        CandidateId("L21_V001", -1)

    # Rejection of unrendered candidate during refinement
    service = _create_test_service(tmp_path)
    session_id = "session-unrendered-cand"
    service.start_session(session_id, "query")

    unrendered_cid = CandidateId("UNKNOWN_VIDEO", 9999)
    with pytest.raises(FeedbackValidationError, match="candidate is not rendered"):
        service.refine(session_id, unrendered_cid, "refine text", expected_revision=0)


# ---------------------------------------------------------------------------
# 5. Revision Ordering & CAS Concurrency Invariant
# ---------------------------------------------------------------------------


def test_wp08_revision_ordering_and_cas_concurrency(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)
    session_id = "session-cas-test"
    service.start_session(session_id, "query")

    # Correct revision 0 -> 1 succeeds
    v1 = service.refine(session_id, CandidateId("L21_V001", 100), "feedback 1", expected_revision=0)
    assert v1.revision == 1

    # Stale revision 0 -> fails with RevisionConflict
    with pytest.raises(RevisionConflict, match="session revision is stale"):
        service.refine(session_id, CandidateId("L21_V001", 100), "feedback concurrent", expected_revision=0)

    # Correct revision 1 -> 2 succeeds
    v2 = service.refine(session_id, CandidateId("L21_V001", 200), "feedback 2", expected_revision=1)
    assert v2.revision == 2

    # Stale revision on undo
    with pytest.raises(RevisionConflict, match="session revision is stale"):
        service.undo(session_id, expected_revision=1)

    # Correct revision on undo 2 -> 3 succeeds
    v3 = service.undo(session_id, expected_revision=2)
    assert v3.revision == 3


# ---------------------------------------------------------------------------
# 6. Event History Limit & Token Budget Invariants
# ---------------------------------------------------------------------------


def test_wp08_event_history_limit_and_token_budget(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)
    session_id = "session-limits-test"
    service.start_session(session_id, "initial search")

    # Perform exactly 5 feedback events (allowed limit)
    rev = 0
    for i in range(1, 6):
        view = service.refine(session_id, CandidateId("L21_V001", 100), f"step {i}", expected_revision=rev)
        rev = view.revision
        assert rev == i

    # 6th feedback event must be rejected
    with pytest.raises(FeedbackValidationError, match="session permits at most five active feedback events"):
        service.refine(session_id, CandidateId("L21_V001", 100), "step 6 exceeds limit", expected_revision=rev)

    # Empty feedback text rejection
    with pytest.raises(FeedbackValidationError, match="feedback text must contain a non-whitespace character"):
        service.refine(session_id, CandidateId("L21_V001", 100), "", expected_revision=rev)

    with pytest.raises(FeedbackValidationError, match="feedback text must contain a non-whitespace character"):
        service.refine(session_id, CandidateId("L21_V001", 100), "   \t\n  ", expected_revision=rev)

    # Feedback text > 300 characters rejection
    long_text = "a" * 301
    with pytest.raises(FeedbackValidationError, match="feedback text must not exceed 300 raw characters"):
        service.refine(session_id, CandidateId("L21_V001", 100), long_text, expected_revision=rev)

    # Token budget validation
    with pytest.raises(FeedbackValidationError, match="feedback template exceeds 5 tokens"):
        validate_token_budget("one two three four five six", token_counter=lambda s: len(s.split()), limit=5)


# ---------------------------------------------------------------------------
# 7. Undo & Reset Semantics
# ---------------------------------------------------------------------------


def test_wp08_undo_and_reset_semantics(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)
    session_id = "session-undo-reset"
    original_query = "crowded market morning"
    initial_view = service.start_session(session_id, original_query)
    initial_order = tuple(c.candidate_id for c in initial_view.candidates)

    # Refine 1
    v1 = service.refine(session_id, CandidateId("L21_V002", 300), "fruit stall", expected_revision=0)
    assert tuple(c.candidate_id for c in v1.candidates)[0] == CandidateId("L21_V002", 300)

    # Refine 2
    v2 = service.refine(session_id, CandidateId("L21_V003", 400), "yellow canopy", expected_revision=1)
    assert tuple(c.candidate_id for c in v2.candidates)[0] == CandidateId("L21_V003", 400)

    # Undo -> pops "yellow canopy", reranks based on "fruit stall"
    v3 = service.undo(session_id, expected_revision=2)
    assert v3.revision == 3
    assert tuple(c.candidate_id for c in v3.candidates)[0] == CandidateId("L21_V002", 300)

    # Reset -> restores initial unrefined order, clears events
    v4 = service.reset(session_id, expected_revision=3)
    assert v4.revision == 4
    assert tuple(c.candidate_id for c in v4.candidates) == initial_order

    # Inspect state to verify events cleared and original_query intact
    _, _, _, state = service._store.get_record(session_id)
    assert state["events"] == []
    assert state["original_query"] == original_query


# ---------------------------------------------------------------------------
# 8. Separate Snapshots: Initial C0 vs Refined Results
# ---------------------------------------------------------------------------


def test_wp08_separate_snapshots_initial_vs_refined(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)
    session_id = "session-snapshots"
    service.start_session(session_id, "query")

    service.refine(session_id, CandidateId("L21_V002", 300), "refine 1", expected_revision=0)

    _, _, _, state = service._store.get_record(session_id)
    assert "initial" in state
    assert "initial_rendered" in state
    assert "rendered" in state

    # Initial vs refined rendered lists are distinct
    initial_rendered = [CandidateId(item["video_id"], item["frame_id"]) for item in state["initial_rendered"]]
    refined_rendered = [CandidateId(item["video_id"], item["frame_id"]) for item in state["rendered"]]
    assert initial_rendered[0] == CandidateId("L21_V001", 100)
    assert refined_rendered[0] == CandidateId("L21_V002", 300)
    assert initial_rendered != refined_rendered

    # Initial pool C0 is preserved in full
    initial_c0 = [CandidateId(item["video_id"], item["frame_id"]) for item in state["initial"]]
    assert set(refined_rendered).issubset(set(initial_c0))


# ---------------------------------------------------------------------------
# 9. Basket Non-Mutation Invariant
# ---------------------------------------------------------------------------


def test_wp08_basket_non_mutation_invariant(tmp_path: Path) -> None:
    """Feedback operations MUST NEVER mutate the operator's submission basket."""
    service = _create_test_service(tmp_path)
    session_id = "session-basket-isolation"

    # Simulated operator basket
    submission_basket = [
        {"video_id": "L21_V001", "frame_id": 100, "approved_answer": None},
        {"video_id": "L21_V099", "frame_id": 999, "approved_answer": None},
    ]
    basket_before = list(submission_basket)

    # Perform feedback operations
    service.start_session(session_id, "search query")
    assert submission_basket == basket_before

    service.refine(session_id, CandidateId("L21_V001", 100), "feedback text", expected_revision=0)
    assert submission_basket == basket_before

    service.undo(session_id, expected_revision=1)
    assert submission_basket == basket_before

    service.reset(session_id, expected_revision=2)
    assert submission_basket == basket_before

    service.confirm(session_id, CandidateId("L21_V001", 100), expected_revision=3)
    assert submission_basket == basket_before

    service.view(session_id)
    assert submission_basket == basket_before


# ---------------------------------------------------------------------------
# 10. Unavailable and Expired Sessions & Error Propagation
# ---------------------------------------------------------------------------


def test_wp08_unavailable_and_expired_sessions(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)

    # Nonexistent session raises SessionExpired("session is unavailable")
    with pytest.raises(SessionExpired, match="session is unavailable"):
        service.view("nonexistent-session")

    with pytest.raises(SessionExpired, match="session is unavailable"):
        service.refine("nonexistent-session", CandidateId("L21_V001", 100), "text", 0)

    # Expired session
    now = _fixed_clock(0)
    past_clock = lambda: now
    expired_service = _create_test_service(tmp_path, clock_fn=past_clock)
    expired_service.start_session("expiring-session", "query")

    # Fast forward clock beyond 24 hours
    future_clock = lambda: now + timedelta(hours=25)
    future_service = _create_test_service(tmp_path, clock_fn=future_clock)

    with pytest.raises(SessionExpired, match="session has expired"):
        future_service.view("expiring-session")


def test_wp08_model_ranking_failure_fails_closed(tmp_path: Path) -> None:
    def failing_ranker(candidates, template, selected, snapshot):
        raise RuntimeError("GPU OOM / model failure")

    service = _create_test_service(tmp_path, ranker=failing_ranker)
    session_id = "session-ranker-fail"
    service.start_session(session_id, "query")

    with pytest.raises(ModelRankingFailed, match="four-model feedback ranking failed"):
        service.refine(session_id, CandidateId("L21_V001", 100), "text", expected_revision=0)

    # Session revision remains untouched at 0 after failed refinement
    view = service.view(session_id)
    assert view.revision == 0


# ---------------------------------------------------------------------------
# 11. Confirmation & First-Correct Metrics
# ---------------------------------------------------------------------------


def test_wp08_confirmation_and_first_correct_metrics(tmp_path: Path) -> None:
    service = _create_test_service(tmp_path)
    session_id = "session-metrics"
    service.start_session(session_id, "query")

    # Confirmation is idempotent and does not advance ranking revision
    c1 = service.confirm(session_id, CandidateId("L21_V001", 100), expected_revision=0)
    assert c1.session_id == session_id
    assert c1.revision == 0
    assert c1.candidate_id == CandidateId("L21_V001", 100)

    c2 = service.confirm(session_id, CandidateId("L21_V001", 100), expected_revision=0)
    assert c2.confirmation_id == c1.confirmation_id

    # First-correct recording: cohort is 'no_feedback' when events is empty
    fc_no_fb = service.record_correct(session_id, CandidateId("L21_V001", 100), expected_revision=0)
    assert fc_no_fb.cohort == "no_feedback"

    # Start second session and record with feedback
    s2 = "session-metrics-fb"
    service.start_session(s2, "query 2")
    service.refine(s2, CandidateId("L21_V001", 100), "refined", expected_revision=0)
    fc_with_fb = service.record_correct(s2, CandidateId("L21_V001", 100), expected_revision=1)
    assert fc_with_fb.cohort == "with_feedback"

    # Summary metrics aggregation
    summaries = service.feedback_metrics()
    summary_map = {s.cohort: s.count for s in summaries}
    assert summary_map["no_feedback"] == 1
    assert summary_map["with_feedback"] == 1


# ---------------------------------------------------------------------------
# 12. TV4 Feedback Seam Verification (T020 Integrated Endpoints)
# ---------------------------------------------------------------------------


def test_tv4_feedback_seam_endpoints_exposed() -> None:
    """Verifies that TV4 exposes the required /feedback endpoints integrated in T020."""
    tv4_api_path = TV4_SRC / "tv4" / "api.py"
    assert tv4_api_path.exists(), "tv4/api.py must exist"
    content = tv4_api_path.read_text(encoding="utf-8")

    # Characterize existing base endpoints
    assert '@app.get("/health")' in content
    assert '@app.post("/kis/search")' in content
    assert '@app.post("/vqa/answer")' in content
    assert '@app.post("/trake/align")' in content
    assert '@app.post("/exact-frame/neighbors")' in content

    # Verify T020 integrated feedback endpoints
    assert '@app.post("/feedback/start")' in content
    assert '@app.post("/feedback/refine")' in content
    assert '@app.post("/feedback/undo")' in content
    assert '@app.post("/feedback/reset")' in content
    assert '@app.get("/feedback/session/{session_id}")' in content
