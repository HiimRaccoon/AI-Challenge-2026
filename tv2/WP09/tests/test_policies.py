from __future__ import annotations

from wp09.decoder import DecodedFrame
from wp09.policies import ScoredFrame, select_hypotheses
from wp09.contracts import RefinementPolicy, Task


def test_transition_policy_prefers_largest_adjacent_score_change() -> None:
    """Catches treating a TRAKE transition as merely the highest-scoring frame."""

    frames = (
        ScoredFrame(DecodedFrame(10, 10, "1/100", 100), visual_score=0.1),
        ScoredFrame(DecodedFrame(11, 11, "1/100", 110), visual_score=0.2),
        ScoredFrame(DecodedFrame(12, 12, "1/100", 120), visual_score=0.9),
    )

    chosen = select_hypotheses(Task.TRAKE, RefinementPolicy.TRANSITION, frames, limit=1)

    assert chosen[0].frame.frame_id == 12
