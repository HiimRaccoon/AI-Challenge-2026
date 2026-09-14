from __future__ import annotations

from datetime import UTC, datetime, timedelta

from wp08.contracts import CandidateId, CandidateMetadata, SessionPool
from wp08.service import FeedbackSessions
from wp08.store import SqliteSessionStore


def test_refine_undo_reset_and_confirm_keep_revision_semantics(tmp_path) -> None:
    now = lambda: datetime(2026, 8, 9, tzinfo=UTC)
    service = FeedbackSessions(
        store=SqliteSessionStore(tmp_path / "sessions.db", clock=now),
        clock=now,
        pool_provider=lambda _: SessionPool(
            wp03_run_id="run-1",
            candidates=(CandidateId("L21_V001", 42), CandidateId("L21_V002", 84)),
            candidate_metadata=(
                CandidateMetadata(CandidateId("L21_V001", 42), 42, "one.jpg"),
                CandidateMetadata(CandidateId("L21_V002", 84), 84, "two.jpg"),
            ),
            snapshot={"fixture": True},
            provenance={},
        ),
        ranker=lambda candidates, _text, _selected, _snapshot: tuple(reversed(candidates)),
        token_counter=lambda text: len(text.split()),
    )
    initial = service.start_session("s-1", "a bus stop")
    refined = service.refine("s-1", CandidateId("L21_V001", 42), "at night", 0)
    assert refined.revision == 1
    undone = service.undo("s-1", 1)
    assert undone.revision == 2
    reset = service.reset("s-1", 2)
    assert reset.revision == 3
    assert service.confirm("s-1", CandidateId("L21_V001", 42), 3).revision == 3
    measured = service.record_correct("s-1", CandidateId("L21_V001", 42), 3)
    assert measured.cohort == "no_feedback"
