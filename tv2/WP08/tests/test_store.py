from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wp08.contracts import CandidateId, RevisionConflict, SessionExpired
from wp08.store import SqliteSessionStore


def now() -> datetime:
    return datetime(2026, 8, 9, tzinfo=UTC)


def test_store_rejects_stale_revision_and_deduplicates_confirmation(tmp_path) -> None:
    store = SqliteSessionStore(tmp_path / "sessions.db", clock=now)
    store.create(session_id="s-1", wp03_run_id="run-1", expires_at=now() + timedelta(hours=24), state={"rank": 0})
    assert store.commit(session_id="s-1", expected_revision=0, state={"rank": 1}, reason="refine") == 1
    with pytest.raises(RevisionConflict):
        store.commit(session_id="s-1", expected_revision=0, state={"rank": 2}, reason="refine")
    first = store.confirm(session_id="s-1", expected_revision=1, candidate_id=CandidateId("L21_V001", 42))
    second = store.confirm(session_id="s-1", expected_revision=1, candidate_id=CandidateId("L21_V001", 42))
    assert second.confirmation_id == first.confirmation_id


def test_store_expires_session_and_releases_artifact_lease(tmp_path) -> None:
    expired = now() - timedelta(seconds=1)
    store = SqliteSessionStore(tmp_path / "sessions.db", clock=now)
    store.create(session_id="s-1", wp03_run_id="run-1", expires_at=expired, state={})
    with pytest.raises(SessionExpired):
        store.get("s-1")
    assert store.active_artifact_runs(now()) == set()


def test_store_records_first_correct_once_and_groups_metrics(tmp_path) -> None:
    store = SqliteSessionStore(tmp_path / "sessions.db", clock=now)
    store.create(session_id="s-1", wp03_run_id="run-1", expires_at=now() + timedelta(hours=24), state={})
    first = store.record_first_correct(session_id="s-1", expected_revision=0, candidate_id=CandidateId("L21_V001", 42), cohort="no_feedback")
    repeated = store.record_first_correct(session_id="s-1", expected_revision=0, candidate_id=CandidateId("L21_V002", 84), cohort="with_feedback")
    assert repeated == first
    assert store.feedback_metrics()[0].count == 1
    assert store.feedback_metrics()[1].count == 0
