"""Deterministic task policies for local original-frame scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean, pvariance
from typing import Sequence

from .contracts import ContractError, EvidenceContribution, RefinementPolicy, Task
from .decoder import DecodedFrame


@dataclass(frozen=True)
class ScoredFrame:
    frame: DecodedFrame
    visual_score: float
    quality_score: float = 0.0
    evidence: tuple[EvidenceContribution, ...] = ()
    policy_score: float = 0.0
    policy_target_pts: int | None = None

    @property
    def evidence_score(self) -> float:
        return sum(item.score for item in self.evidence)

    @property
    def score(self) -> float:
        return self.visual_score + self.quality_score + self.evidence_score


def select_hypotheses(task: Task, policy: RefinementPolicy, frames: Sequence[ScoredFrame], *, limit: int, stable_variance_penalty: float = 0.25) -> tuple[ScoredFrame, ...]:
    """Policy-score and total-order exact-frame suggestions.

    All returned frames share the target PTS selected by the policy, so tie
    breaking is independent of incidental decoder order.
    """
    if not frames or limit <= 0:
        raise ContractError("hypothesis selection requires frames and positive limit")
    _validate_task_policy(task, policy)
    ordered = tuple(sorted(frames, key=lambda item: (item.frame.timestamp_ms, item.frame.frame_id)))
    if policy is RefinementPolicy.TRANSITION:
        enriched, target = _transition(ordered)
    elif policy is RefinementPolicy.PEAK:
        enriched, target = _peak(ordered)
    elif policy is RefinementPolicy.STABLE_STATE:
        enriched, target = _stable(ordered, stable_variance_penalty)
    elif policy is RefinementPolicy.EVIDENCE_VISIBLE:
        enriched, target = _standard(ordered, evidence_first=True)
    else:
        enriched, target = _standard(ordered, evidence_first=False)
    total_order = sorted(
        enriched,
        key=lambda item: (-item.policy_score, -item.visual_score, abs(item.frame.pts - target), item.frame.timestamp_ms, item.frame.frame_id),
    )
    return tuple(total_order[:limit])


def _standard(frames: Sequence[ScoredFrame], *, evidence_first: bool) -> tuple[tuple[ScoredFrame, ...], int]:
    # Evidence already contributes once through ScoredFrame.score. VQA changes
    # selection purpose, not arithmetic weight.
    score = lambda item: item.score
    target_item = max(frames, key=lambda item: (score(item), item.visual_score, -item.frame.timestamp_ms, -item.frame.frame_id))
    return tuple(replace(item, policy_score=score(item), policy_target_pts=target_item.frame.pts) for item in frames), target_item.frame.pts


def _transition(frames: Sequence[ScoredFrame]) -> tuple[tuple[ScoredFrame, ...], int]:
    values: list[float] = [0.0]
    for left, right in zip(frames, frames[1:]):
        delta_ms = max(1, right.frame.timestamp_ms - left.frame.timestamp_ms)
        values.append(abs(right.score - left.score) / delta_ms)
    best_index = max(range(len(frames)), key=lambda index: (values[index], frames[index].visual_score, -frames[index].frame.timestamp_ms, -frames[index].frame.frame_id))
    target = frames[best_index].frame.pts
    return tuple(replace(item, policy_score=values[index], policy_target_pts=target) for index, item in enumerate(frames)), target


def _peak(frames: Sequence[ScoredFrame]) -> tuple[tuple[ScoredFrame, ...], int]:
    values = [item.score for item in frames]
    local = [value if (i == 0 or value >= values[i - 1]) and (i == len(values) - 1 or value >= values[i + 1]) else value - 1.0 for i, value in enumerate(values)]
    best_index = max(range(len(frames)), key=lambda index: (local[index], frames[index].visual_score, -frames[index].frame.timestamp_ms, -frames[index].frame.frame_id))
    target = frames[best_index].frame.pts
    return tuple(replace(item, policy_score=local[index], policy_target_pts=target) for index, item in enumerate(frames)), target


def _stable(frames: Sequence[ScoredFrame], penalty: float) -> tuple[tuple[ScoredFrame, ...], int]:
    if penalty < 0:
        raise ContractError("stable variance penalty must be non-negative")
    values: list[float] = []
    for index, item in enumerate(frames):
        local = [entry.score for entry in frames[max(0, index - 1): min(len(frames), index + 2)]]
        values.append(fmean(local) - penalty * (pvariance(local) if len(local) > 1 else 0.0))
    best_index = max(range(len(frames)), key=lambda index: (values[index], frames[index].visual_score, -frames[index].frame.timestamp_ms, -frames[index].frame.frame_id))
    target = frames[best_index].frame.pts
    return tuple(replace(item, policy_score=values[index], policy_target_pts=target) for index, item in enumerate(frames)), target


def _validate_task_policy(task: Task, policy: RefinementPolicy) -> None:
    expected = {
        Task.KIS: {RefinementPolicy.REPRESENTATIVE},
        Task.VQA: {RefinementPolicy.EVIDENCE_VISIBLE},
        Task.TRAKE: {RefinementPolicy.TRANSITION, RefinementPolicy.PEAK, RefinementPolicy.STABLE_STATE},
    }
    if policy not in expected.get(task, set()):
        raise ContractError(f"{task.value} does not accept {policy.value} policy")
