"""State-machine orchestration for an interactive feedback session."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta

from .contracts import CandidateId, CandidateMetadata, Confirmation, FeedbackEvent, FeedbackMetricSummary, FeedbackValidationError, FirstCorrect, ModelRankingFailed, RenderedCandidate, SessionPool, SessionView
from .store import SqliteSessionStore
from .text import build_feedback_template, validate_token_budget


class FeedbackSessions:
    def __init__(
        self,
        *,
        store: SqliteSessionStore,
        clock: Callable[[], datetime],
        pool_provider: Callable[[str], SessionPool],
        ranker: Callable[[Sequence[CandidateId], str, CandidateId | None, Mapping[str, object]], Sequence[CandidateId]],
        token_counter: Callable[[str], int],
        renderer: Callable[[Sequence[CandidateId], Mapping[str, object]], Sequence[CandidateId]] | None = None,
    ) -> None:
        self._store, self._clock = store, clock
        self._pool_provider, self._ranker, self._token_counter = pool_provider, ranker, token_counter
        self._renderer = renderer or (lambda candidates, _snapshot: tuple(candidates)[:100])

    def start_session(self, session_id: str, original_query: str) -> SessionView:
        if not isinstance(original_query, str) or not original_query.strip():
            raise FeedbackValidationError("original query must be non-empty")
        pool = self._pool_provider(original_query)
        candidates = tuple(pool.candidates)
        rendered = self._render(candidates, pool.snapshot)
        state = {
            "original_query": original_query,
            "initial": [self._encode(item) for item in candidates],
            "rendered": [self._encode(item) for item in rendered],
            "initial_rendered": [self._encode(item) for item in rendered],
            "events": [],
            "snapshot": dict(pool.snapshot),
            "provenance": dict(pool.provenance),
            "candidate_metadata": [self._encode_metadata(item) for item in pool.candidate_metadata],
        }
        self._store.create(session_id=session_id, wp03_run_id=pool.wp03_run_id, expires_at=self._clock() + timedelta(hours=24), state=state)
        return self.view(session_id)

    def refine(self, session_id: str, selected: CandidateId, feedback_text: str, expected_revision: int) -> SessionView:
        revision, state = self._store.get(session_id)
        if revision != expected_revision:
            from .contracts import RevisionConflict
            raise RevisionConflict("session revision is stale")
        if selected not in {self._decode(item) for item in state["rendered"]}:
            raise FeedbackValidationError("candidate is not rendered")
        event = FeedbackEvent.create(candidate_id=selected, feedback_text=feedback_text)
        events = [*state["events"], {"candidate": self._encode(event.candidate_id), "text": event.feedback_text}]
        if len(events) > 5:
            raise FeedbackValidationError("session permits at most five active feedback events")
        parsed = tuple(FeedbackEvent.create(candidate_id=self._decode(item["candidate"]), feedback_text=item["text"]) for item in events)
        template = build_feedback_template(str(state["original_query"]), parsed)
        validate_token_budget(template, token_counter=self._token_counter)
        ranked = self._rerank(state, template, selected)
        rendered = self._render(ranked, self._snapshot(state))
        next_state = {**state, "events": events, "rendered": [self._encode(item) for item in rendered]}
        self._store.commit(session_id=session_id, expected_revision=revision, state=next_state, reason="refine")
        return self.view(session_id)

    def undo(self, session_id: str, expected_revision: int) -> SessionView:
        revision, state = self._store.get(session_id)
        if revision != expected_revision:
            from .contracts import RevisionConflict
            raise RevisionConflict("session revision is stale")
        events = list(state["events"])
        if events:
            events.pop()
        if not events:
            ranked = tuple(self._decode(item) for item in state["initial"])
        else:
            parsed = tuple(FeedbackEvent.create(candidate_id=self._decode(item["candidate"]), feedback_text=item["text"]) for item in events)
            ranked = self._rerank(state, build_feedback_template(str(state["original_query"]), parsed), parsed[-1].candidate_id)
        rendered = self._render(ranked, self._snapshot(state))
        next_state = {**state, "events": events, "rendered": [self._encode(item) for item in rendered]}
        self._store.commit(session_id=session_id, expected_revision=revision, state=next_state, reason="undo")
        return self.view(session_id)

    def reset(self, session_id: str, expected_revision: int) -> SessionView:
        revision, state = self._store.get(session_id)
        if revision != expected_revision:
            from .contracts import RevisionConflict
            raise RevisionConflict("session revision is stale")
        ranked = tuple(self._decode(item) for item in state["initial"])
        rendered = tuple(self._decode(item) for item in state["initial_rendered"])
        next_state = {**state, "events": [], "rendered": [self._encode(item) for item in rendered]}
        self._store.commit(session_id=session_id, expected_revision=expected_revision, state=next_state, reason="reset")
        return self.view(session_id)

    def confirm(self, session_id: str, selected: CandidateId, expected_revision: int) -> Confirmation:
        revision, state = self._store.get(session_id)
        if selected not in {self._decode(item) for item in state["rendered"]}:
            raise FeedbackValidationError("candidate is not rendered")
        return self._store.confirm(session_id=session_id, expected_revision=expected_revision, candidate_id=selected)

    def record_correct(self, session_id: str, selected: CandidateId, expected_revision: int) -> FirstCorrect:
        """Persist the external ground-truth/annotator's first correct result."""
        revision, state = self._store.get(session_id)
        if revision != expected_revision:
            from .contracts import RevisionConflict
            raise RevisionConflict("session revision is stale")
        if selected not in {self._decode(item) for item in state["rendered"]}:
            raise FeedbackValidationError("candidate is not rendered")
        cohort = "with_feedback" if state["events"] else "no_feedback"
        return self._store.record_first_correct(
            session_id=session_id,
            expected_revision=expected_revision,
            candidate_id=selected,
            cohort=cohort,
        )

    def feedback_metrics(self) -> tuple[FeedbackMetricSummary, ...]:
        return self._store.feedback_metrics()

    def view(self, session_id: str) -> SessionView:
        run_id, revision, expires_at, state = self._store.get_record(session_id)
        return self._view_from_state(session_id, revision, state, expires_at_utc=expires_at, wp03_run_id=run_id)

    get_session = view

    def _view_from_state(self, session_id: str, revision: int, state: Mapping[str, object], *, expires_at_utc: str | None = None, wp03_run_id: str | None = None) -> SessionView:
        metadata = {item.candidate_id: item for item in (self._decode_metadata(value) for value in state["candidate_metadata"])}
        return SessionView(
            session_id=session_id,
            revision=revision,
            candidates=tuple(
                RenderedCandidate(candidate_id, rank, metadata[candidate_id].timestamp_ms, metadata[candidate_id].keyframe_path)
                for rank, candidate_id in enumerate((self._decode(item) for item in state["rendered"]), 1)
            ),
            expires_at_utc=expires_at_utc,
            wp03_run_id=wp03_run_id,
        )

    def _rerank(self, state: Mapping[str, object], template: str, selected: CandidateId) -> tuple[CandidateId, ...]:
        initial = tuple(self._decode(item) for item in state["initial"])
        try:
            ranked = tuple(self._ranker(initial, template, selected, self._snapshot(state)))
        except FeedbackValidationError:
            raise
        except Exception as exc:
            raise ModelRankingFailed("four-model feedback ranking failed") from exc
        if len(ranked) != len(initial) or set(ranked) != set(initial):
            raise ModelRankingFailed("feedback ranker did not return exactly C0")
        return ranked

    def _render(self, ranked: Sequence[CandidateId], snapshot: Mapping[str, object]) -> tuple[CandidateId, ...]:
        rendered = tuple(self._renderer(ranked, snapshot))
        if len(rendered) > 100 or len(set(rendered)) != len(rendered) or not set(rendered).issubset(set(ranked)):
            raise ModelRankingFailed("renderer returned an invalid display list")
        return rendered

    @staticmethod
    def _snapshot(state: Mapping[str, object]) -> Mapping[str, object]:
        snapshot = state.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise FeedbackValidationError("session lacks its immutable WP03 snapshot")
        return snapshot

    @staticmethod
    def _encode(value: CandidateId) -> dict[str, object]:
        return {"video_id": value.video_id, "frame_id": value.frame_id}

    @staticmethod
    def _decode(value: object) -> CandidateId:
        raw = dict(value)  # type: ignore[arg-type]
        return CandidateId(str(raw["video_id"]), int(raw["frame_id"]))

    @classmethod
    def _encode_metadata(cls, value: CandidateMetadata) -> dict[str, object]:
        return {"candidate": cls._encode(value.candidate_id), "timestamp_ms": value.timestamp_ms, "keyframe_path": value.keyframe_path}

    @classmethod
    def _decode_metadata(cls, value: object) -> CandidateMetadata:
        raw = dict(value)  # type: ignore[arg-type]
        return CandidateMetadata(cls._decode(raw["candidate"]), int(raw["timestamp_ms"]), str(raw["keyframe_path"]))
