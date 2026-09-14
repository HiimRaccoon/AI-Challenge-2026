from __future__ import annotations

from wp09.contracts import RefinementPolicy, Task
from wp09.decoder import DecodedFrame
from wp09.policies import ScoredFrame, select_hypotheses


def test_total_order_uses_all_five_documented_tie_breaks() -> None:
    """Catches non-deterministic suggestions when policy and visual scores tie."""
    frames = (
        ScoredFrame(DecodedFrame(42, 1, "1/1000", 100), visual_score=0.8, policy_score=1.0, policy_target_pts=50),
        ScoredFrame(DecodedFrame(41, 2, "1/1000", 100), visual_score=0.8, policy_score=1.0, policy_target_pts=50),
        ScoredFrame(DecodedFrame(40, 3, "1/1000", 120), visual_score=0.7, policy_score=1.0, policy_target_pts=50),
    )
    chosen = select_hypotheses(Task.TRAKE, RefinementPolicy.PEAK, frames, limit=3)
    assert [item.frame.frame_id for item in chosen] == [41, 42, 40]


def test_transition_normalizes_score_change_by_timestamp_delta() -> None:
    """Catches fast transitions being hidden by a larger but slow score change."""
    frames = (
        ScoredFrame(DecodedFrame(1, 1, "1/1000", 0), 0.0),
        ScoredFrame(DecodedFrame(2, 2, "1/1000", 100), 0.4),
        ScoredFrame(DecodedFrame(3, 3, "1/1000", 1_000), 1.0),
    )
    chosen = select_hypotheses(Task.TRAKE, RefinementPolicy.TRANSITION, frames, limit=1)
    assert chosen[0].frame.frame_id == 2
