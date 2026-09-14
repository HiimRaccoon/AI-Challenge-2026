from __future__ import annotations

import pytest

from wp09.contracts import CoarseCandidate, ContractError, DecodeBudget, FrameSelection, validate_trake_selection


def test_coarse_candidate_rejects_missing_pts_timestamp() -> None:
    """Catches accepting a frame whose original-video time cannot be resolved."""

    with pytest.raises(ContractError, match="timestamp_ms"):
        CoarseCandidate(video_id="L21_V001", frame_id=10, timestamp_ms=None)  # type: ignore[arg-type]


def test_trake_selection_rejects_duplicate_frame_ids() -> None:
    """Catches submitting two ordered TRAKE events at the same semantic frame."""

    with pytest.raises(ContractError, match="strictly increasing"):
        validate_trake_selection((FrameSelection("L21_V001", 100), FrameSelection("L21_V001", 100)))


def test_candidate_keeps_upstream_score_and_confidence_for_audit_and_adaptive_radius() -> None:
    """Catches losing the only upstream signals WP09 may use without re-retrieving."""

    candidate = CoarseCandidate("L21_V001", 10, 57, upstream_score=0.7, confidence=0.4)

    assert candidate.upstream_score == pytest.approx(0.7)
    assert candidate.confidence == pytest.approx(0.4)


def test_trake_selection_rejects_a_cross_video_chain_even_with_order_exception() -> None:
    """Catches an ordering override accidentally allowing the TRAKE wrong-video zero-score case."""

    with pytest.raises(ContractError, match="same video"):
        validate_trake_selection(
            (FrameSelection("L21_V001", 100), FrameSelection("L21_V002", 101)),
            allow_order_exception=True,
        )


def test_decode_budget_limits_frames_window_and_regions() -> None:
    """Catches an unbounded refinement request despite a bounded local window."""

    budget = DecodeBudget(max_decoded_frames=120, max_window_ms=8_000, max_decode_time_ms=1_000, max_dense_regions=2)

    assert budget.max_dense_regions == 2
