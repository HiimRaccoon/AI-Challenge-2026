"""SQLite persistence for immutable snapshots and short CAS mutations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .contracts import CandidateId, Confirmation, FeedbackMetricSummary, FirstCorrect, RevisionConflict, SessionExpired


class SqliteSessionStore:
    def __init__(self, path: Path, *, clock: Callable[[], datetime]) -> None:
        self._path, self._clock = path, clock
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY, wp03_run_id TEXT NOT NULL,
                  created_at TEXT NOT NULL, expires_at TEXT NOT NULL, revision INTEGER NOT NULL, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS confirmations (
                  confirmation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, revision INTEGER NOT NULL,
                  video_id TEXT NOT NULL, frame_id INTEGER NOT NULL, created_at TEXT NOT NULL,
                  UNIQUE(session_id, revision, video_id, frame_id)
                );
                CREATE TABLE IF NOT EXISTS first_correct (
                  session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                  video_id TEXT NOT NULL, frame_id INTEGER NOT NULL,
                  cohort TEXT NOT NULL CHECK(cohort IN ('no_feedback', 'with_feedback')),
                  recorded_at TEXT NOT NULL, elapsed_ms INTEGER NOT NULL
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            if "created_at" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN created_at TEXT")
                # Existing pre-metric sessions retain readability. Their exact
                # creation time was not persisted, so do not fabricate a metric.
                db.execute("UPDATE sessions SET created_at=expires_at WHERE created_at IS NULL")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    @staticmethod
    def _utc(value: datetime) -> str:
        return value.isoformat()

    def _row(self, db: sqlite3.Connection, session_id: str) -> tuple[str, int, str, str]:
        row = db.execute("SELECT wp03_run_id, revision, expires_at, state FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise SessionExpired("session is unavailable")
        if datetime.fromisoformat(row[2]) <= self._clock():
            raise SessionExpired("session has expired")
        return row

    def _created_at(self, db: sqlite3.Connection, session_id: str) -> datetime:
        row = db.execute("SELECT created_at FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None or row[0] is None:
            raise SessionExpired("session is unavailable")
        return datetime.fromisoformat(row[0])

    def create(self, *, session_id: str, wp03_run_id: str, expires_at: datetime, state: dict[str, object]) -> None:
        with self._connect() as db:
            created_at = self._clock()
            db.execute(
                "INSERT INTO sessions (session_id, wp03_run_id, created_at, expires_at, revision, state) VALUES (?, ?, ?, ?, 0, ?)",
                (session_id, wp03_run_id, self._utc(created_at), self._utc(expires_at), json.dumps(state)),
            )

    def get(self, session_id: str) -> tuple[int, dict[str, object]]:
        _, revision, _, state = self.get_record(session_id)
        return revision, state

    def get_record(self, session_id: str) -> tuple[str, int, str, dict[str, object]]:
        """Return persisted immutable provenance plus current mutable state."""
        with self._connect() as db:
            run_id, revision, expires_at, state = self._row(db, session_id)
            return run_id, revision, expires_at, json.loads(state)

    def commit(self, *, session_id: str, expected_revision: int, state: dict[str, object], reason: str) -> int:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, revision, _, _ = self._row(db, session_id)
            if revision != expected_revision:
                raise RevisionConflict("session revision is stale")
            next_revision = revision + 1
            db.execute("UPDATE sessions SET revision=?, state=? WHERE session_id=?", (next_revision, json.dumps({**state, "reason": reason}), session_id))
            return next_revision

    def confirm(self, *, session_id: str, expected_revision: int, candidate_id: CandidateId) -> Confirmation:
        with self._connect() as db:
            _, revision, _, _ = self._row(db, session_id)
            if revision != expected_revision:
                raise RevisionConflict("session revision is stale")
            existing = db.execute("SELECT confirmation_id, created_at FROM confirmations WHERE session_id=? AND revision=? AND video_id=? AND frame_id=?", (session_id, revision, candidate_id.video_id, candidate_id.frame_id)).fetchone()
            if existing is None:
                confirmation_id, created_at = str(uuid.uuid4()), self._utc(self._clock())
                db.execute("INSERT INTO confirmations VALUES (?, ?, ?, ?, ?, ?)", (confirmation_id, session_id, revision, candidate_id.video_id, candidate_id.frame_id, created_at))
            else:
                confirmation_id, created_at = existing
            return Confirmation(confirmation_id, session_id, revision, candidate_id, created_at)

    def active_artifact_runs(self, now: datetime) -> set[str]:
        with self._connect() as db:
            return {row[0] for row in db.execute("SELECT DISTINCT wp03_run_id FROM sessions WHERE expires_at > ?", (self._utc(now),))}

    def record_first_correct(self, *, session_id: str, expected_revision: int, candidate_id: CandidateId, cohort: str) -> FirstCorrect:
        if cohort not in {"no_feedback", "with_feedback"}:
            raise ValueError("feedback cohort is invalid")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, revision, _, _ = self._row(db, session_id)
            if revision != expected_revision:
                raise RevisionConflict("session revision is stale")
            existing = db.execute(
                "SELECT revision, video_id, frame_id, cohort, recorded_at, elapsed_ms FROM first_correct WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if existing is None:
                recorded_at = self._clock()
                elapsed_ms = max(0, int((recorded_at - self._created_at(db, session_id)).total_seconds() * 1_000))
                created = (revision, candidate_id.video_id, candidate_id.frame_id, cohort, self._utc(recorded_at), elapsed_ms)
                db.execute("INSERT INTO first_correct VALUES (?, ?, ?, ?, ?, ?, ?)", (session_id, *created))
                existing = created
            return FirstCorrect(
                session_id=session_id,
                revision=int(existing[0]),
                candidate_id=CandidateId(str(existing[1]), int(existing[2])),
                cohort=str(existing[3]),
                recorded_at_utc=str(existing[4]),
                elapsed_ms=int(existing[5]),
            )

    def feedback_metrics(self) -> tuple[FeedbackMetricSummary, ...]:
        with self._connect() as db:
            rows = db.execute("SELECT cohort, elapsed_ms FROM first_correct ORDER BY cohort, elapsed_ms, session_id").fetchall()
        samples = {"no_feedback": [], "with_feedback": []}
        for cohort, elapsed_ms in rows:
            samples[str(cohort)].append(int(elapsed_ms))
        return tuple(FeedbackMetricSummary(cohort, len(samples[cohort]), tuple(samples[cohort])) for cohort in ("no_feedback", "with_feedback"))
