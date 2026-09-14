"""TV4 adapter for WP08 Interactive Composed Feedback service.

Translates between TV4 API requests/responses and WP08 FeedbackSessions
state-machine contracts. Preserves canonical CandidateId identity, optimistic
CAS revision concurrency, original query immutability, and separate snapshot
semantics.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure WP08 tracked and runtime sources can be loaded
_CANDIDATE_WP08_ROOTS = (
    Path(__file__).resolve().parents[4] / "tv2_1" / "WP08" / "src",        # runtime WP08 when in runtime tv4
    Path(__file__).resolve().parents[4] / "TV2" / "WP08" / "src",          # tracked WP08 when in tv5/tv4
    Path(__file__).resolve().parents[5] / "tv2_1" / "WP08" / "src",        # runtime WP08 when in tv5/tv4
    Path(__file__).resolve().parents[4] / "tv5" / "TV2" / "WP08" / "src",  # tracked WP08 when in runtime tv4
)

for _src in _CANDIDATE_WP08_ROOTS:
    if _src.exists() and str(_src.resolve()) not in sys.path:
        sys.path.insert(0, str(_src.resolve()))

from wp08.contracts import (
    CandidateId,
    CandidateMetadata,
    FeedbackValidationError,
    ModelRankingFailed,
    RevisionConflict,
    SessionExpired,
    SessionPool,
    SessionView,
)
from wp08.service import FeedbackSessions
from wp08.store import SqliteSessionStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_fixture_pool(query: str, preprocess_run_id: str = "run_v1_batch1") -> SessionPool:
    """Deterministic fixture candidates matching TV4 fixture dataset."""
    cands = [
        ("L21_V001", 10690, 356333, "keyframes/L21_V001/83.jpg"),
        ("L21_V002", 23940, 798000, "keyframes/L21_V002/214.jpg"),
        ("L05_V005", 888, 29600, "keyframes/L05_V005/0888.jpg"),
        ("L10_V010", 101, 3366, "keyframes/L10_V010/0101.jpg"),
    ]
    cid_list = tuple(CandidateId(vid, fid) for vid, fid, _, _ in cands)
    meta_list = tuple(
        CandidateMetadata(cid, ts, path)
        for (cid, (_, _, ts, path)) in zip(cid_list, cands)
    )
    return SessionPool(
        wp03_run_id=preprocess_run_id,
        candidates=cid_list,
        candidate_metadata=meta_list,
        snapshot={"fixture": True, "query": query, "pool_size": len(cid_list)},
        provenance={"mode": "fixture", "model": "fixture_reranker", "rrf_k": 60},
    )


def _default_fixture_ranker(
    candidates: Sequence[CandidateId],
    template: str,
    selected: CandidateId | None,
    snapshot: Mapping[str, object],
) -> Sequence[CandidateId]:
    """Deterministic fixture ranker: moves selected candidate to top, retains C0."""
    cands = list(candidates)
    if selected and selected in cands:
        cands.remove(selected)
        return (selected, *cands)
    return tuple(cands)


class Wp08FeedbackAdapter:
    """Wraps WP08 FeedbackSessions for TV4 integration."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        clock: Callable[[], datetime] | None = None,
        pool_provider: Callable[[str], SessionPool] | None = None,
        ranker: Callable[[Sequence[CandidateId], str, CandidateId | None, Mapping[str, object]], Sequence[CandidateId]] | None = None,
        token_counter: Callable[[str], int] | None = None,
        renderer: Callable[[Sequence[CandidateId], Mapping[str, object]], Sequence[CandidateId]] | None = None,
        fixture_mode: bool = False,
        preprocess_run_id: str = "run_v1_batch1",
        selectable: bool = False,
    ) -> None:
        self._fixture_mode = fixture_mode
        self._preprocess_run_id = preprocess_run_id
        self._selectable = selectable and not fixture_mode
        self._clock = clock or _utc_now
        self._fixture_db_path: Path | None = None

        if fixture_mode:
            self._fixture_db_path = Path(tempfile.gettempdir()) / f"tv4_feedback_fixture_{uuid.uuid4().hex}.db"
            store_path = self._fixture_db_path
            provider = pool_provider or (lambda q: _make_fixture_pool(q, preprocess_run_id))
            active_ranker = ranker or _default_fixture_ranker
            t_counter = token_counter or (lambda text: len(text.split()))
        else:
            if db_path is None:
                db_path = Path("data/feedback_sessions.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            store_path = db_path
            if pool_provider is None:
                raise FeedbackValidationError("live pool_provider is required in non-fixture mode")
            provider = pool_provider
            active_ranker = ranker or _default_fixture_ranker
            t_counter = token_counter or (lambda text: len(text.split()))

        self._store = SqliteSessionStore(store_path, clock=self._clock)
        self._service = FeedbackSessions(
            store=self._store,
            clock=self._clock,
            pool_provider=provider,
            ranker=active_ranker,
            token_counter=t_counter,
            renderer=renderer,
        )

    def start_session(self, session_id: str, original_query: str) -> dict[str, Any]:
        """Start a new feedback session with immutable original query."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise FeedbackValidationError("session_id must be non-empty string")
        if not isinstance(original_query, str) or not original_query.strip():
            raise FeedbackValidationError("original_query must be non-empty string")

        view = self._service.start_session(session_id.strip(), original_query.strip())
        return self._format_view(view)

    def refine(
        self,
        session_id: str,
        video_id: str,
        frame_id: int,
        feedback_text: str,
        expected_revision: int,
        source_candidate_frame_id: int | None = None,
    ) -> dict[str, Any]:
        """Refine session candidate list based on selected canonical reference frame."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise FeedbackValidationError("session_id must be non-empty string")
        if not isinstance(video_id, str) or not video_id.strip():
            raise FeedbackValidationError("video_id must be non-empty string")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise FeedbackValidationError("frame_id must be non-negative integer")
        if source_candidate_frame_id is not None:
            if isinstance(source_candidate_frame_id, bool) or not isinstance(source_candidate_frame_id, int) or source_candidate_frame_id < 0:
                raise FeedbackValidationError("source_candidate_frame_id must be non-negative integer")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise FeedbackValidationError("expected_revision must be non-negative integer")

        source_frame = source_candidate_frame_id if source_candidate_frame_id is not None else frame_id
        selected = CandidateId(video_id.strip(), source_frame)
        view = self._service.refine(
            session_id=session_id.strip(),
            selected=selected,
            feedback_text=feedback_text,
            expected_revision=expected_revision,
        )
        exact_ref = (video_id.strip(), frame_id, source_frame) if source_candidate_frame_id is not None else None
        return self._format_view(view, exact_reference=exact_ref)

    def undo(self, session_id: str, expected_revision: int) -> dict[str, Any]:
        """Undo last feedback event and revert to previous ranking."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise FeedbackValidationError("session_id must be non-empty string")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise FeedbackValidationError("expected_revision must be non-negative integer")

        view = self._service.undo(session_id=session_id.strip(), expected_revision=expected_revision)
        return self._format_view(view)

    def reset(self, session_id: str, expected_revision: int) -> dict[str, Any]:
        """Reset active view to unrefined initial candidate pool, clearing active events."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise FeedbackValidationError("session_id must be non-empty string")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise FeedbackValidationError("expected_revision must be non-negative integer")

        view = self._service.reset(session_id=session_id.strip(), expected_revision=expected_revision)
        return self._format_view(view)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Retrieve current session view without mutation."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise FeedbackValidationError("session_id must be non-empty string")

        view = self._service.view(session_id.strip())
        return self._format_view(view)

    def _format_view(
        self,
        view: SessionView,
        exact_reference: tuple[str, int, int] | None = None,
    ) -> dict[str, Any]:
        """Format and validate SessionView into TV4 contract shape."""
        if len(view.candidates) > 100:
            raise ModelRankingFailed("WP08 returned more than 100 candidates")

        formatted_candidates = []
        for rank_idx, cand in enumerate(view.candidates, start=1):
            cid = cand.candidate_id
            if not cid.video_id or cid.frame_id < 0:
                raise ModelRankingFailed("WP08 returned non-canonical candidate identity")

            cand_fid = cid.frame_id
            root_anchor_fid = cid.frame_id
            anchor_offset = 0
            if exact_reference is not None:
                ref_vid, ref_fid, ref_source_fid = exact_reference
                if cid.video_id == ref_vid and cid.frame_id == ref_source_fid:
                    cand_fid = ref_fid
                    root_anchor_fid = ref_source_fid
                    anchor_offset = ref_fid - ref_source_fid

            formatted_candidates.append(
                {
                    "video_id": cid.video_id,
                    "frame_id": cand_fid,
                    "certified_anchor_frame_id": root_anchor_fid,
                    "anchor_offset": anchor_offset,
                    "rank": cand.display_rank if cand.display_rank > 0 else rank_idx,
                    "timestamp_ms": cand.timestamp_ms,
                    "keyframe_path": cand.keyframe_path,
                    "submission_selectable": self._selectable,
                    "provenance_mode": "fixture" if self._fixture_mode else "live",
                    "source": "feedback",
                    "preprocess_run_id": view.wp03_run_id or self._preprocess_run_id,
                }
            )

        active_feedback_count = 0
        try:
            _, state = self._store.get(view.session_id)
            active_feedback_count = len(state.get("events", []))
        except Exception:
            pass

        return {
            "session_id": view.session_id,
            "revision": view.revision,
            "candidates": formatted_candidates,
            "active_feedback_count": active_feedback_count,
            "max_active_feedback_events": 5,
            "expires_at_utc": view.expires_at_utc,
            "wp03_run_id": view.wp03_run_id or self._preprocess_run_id,
            "status": "ok",
            "provenance_mode": "fixture" if self._fixture_mode else "live",
        }
