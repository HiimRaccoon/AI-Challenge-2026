"""Exact competition metrics calculation for KIS, VQA, TRAKE, R@k, and Final Score."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from tv5.submission.contracts import KisPrediction, VqaPrediction, TrakePrediction


@dataclass(frozen=True)
class GroundTruthInterval:
    video_id: str
    start_frame_id: int
    end_frame_id: int

    def contains(self, video_id: str, frame_id: int) -> bool:
        return self.video_id == video_id and (self.start_frame_id <= frame_id <= self.end_frame_id)


@dataclass(frozen=True)
class EvaluationResult:
    task_type: str
    query_id: str
    r_score: float
    status: str = "COMPLETED"
    is_hit: bool = False
    details: dict[str, object] = field(default_factory=dict)
    notes: str = ""


def evaluate_kis_prediction(
    prediction: KisPrediction,
    ground_truth: Sequence[GroundTruthInterval],
    query_id: str = "kis",
) -> EvaluationResult:
    """Evaluate KIS prediction: R-score is 1.0 if (video_id, frame_id) is in any accepted interval, else 0.0."""
    matched = any(gt.contains(prediction.video_id, prediction.frame_id) for gt in ground_truth)
    score = 1.0 if matched else 0.0
    return EvaluationResult(
        task_type="KIS",
        query_id=query_id,
        r_score=score,
        is_hit=matched,
        details={"video_id": prediction.video_id, "frame_id": prediction.frame_id},
    )


def evaluate_vqa_prediction(
    prediction: VqaPrediction,
    ground_truth: Sequence[GroundTruthInterval],
    semantic_adjudicator: Callable[[str], bool] | None = None,
    query_id: str = "vqa",
) -> EvaluationResult:
    """Evaluate VQA: requires correct video/interval AND semantic agreement.

    If no authoritative semantic adjudicator is supplied, returns INCOMPLETE status.
    """
    frame_matched = any(gt.contains(prediction.video_id, prediction.frame_id) for gt in ground_truth)
    if not frame_matched:
        return EvaluationResult(
            task_type="VQA",
            query_id=query_id,
            r_score=0.0,
            is_hit=False,
            status="COMPLETED",
            details={"frame_matched": False, "semantic_matched": False},
        )

    if semantic_adjudicator is None:
        return EvaluationResult(
            task_type="VQA",
            query_id=query_id,
            r_score=0.0,
            is_hit=False,
            status="INCOMPLETE / EXTERNAL ADJUDICATION REQUIRED",
            notes="VQA semantic correctness cannot be guessed by exact-string equality; authoritative semantic adjudicator required.",
            details={"frame_matched": True, "semantic_adjudicator_present": False},
        )

    is_semantic_correct = semantic_adjudicator(prediction.approved_answer)
    score = 1.0 if is_semantic_correct else 0.0
    return EvaluationResult(
        task_type="VQA",
        query_id=query_id,
        r_score=score,
        is_hit=is_semantic_correct,
        status="COMPLETED",
        details={"frame_matched": True, "semantic_matched": is_semantic_correct},
    )


def evaluate_trake_prediction(
    prediction: TrakePrediction,
    ground_truth_events: Sequence[GroundTruthInterval],
    query_id: str = "trake",
) -> EvaluationResult:
    """Evaluate TRAKE: If wrong video => score = 0. Otherwise, count of ordered predicted event frames in accepted intervals / N."""
    n_events = len(ground_truth_events)
    if n_events == 0:
        return EvaluationResult(task_type="TRAKE", query_id=query_id, r_score=0.0, status="ERROR", notes="Empty ground truth")

    # If any event GT has a different video, the target video is ground_truth_events[0].video_id
    target_video = ground_truth_events[0].video_id
    if prediction.video_id != target_video:
        return EvaluationResult(
            task_type="TRAKE",
            query_id=query_id,
            r_score=0.0,
            is_hit=False,
            status="COMPLETED",
            details={"wrong_video": True, "predicted_video": prediction.video_id, "target_video": target_video},
        )

    # Check each event frame in sequence
    matched_count = 0
    event_matches: list[bool] = []
    for fid, gt in zip(prediction.event_frame_ids, ground_truth_events):
        m = gt.contains(prediction.video_id, fid)
        event_matches.append(m)
        if m:
            matched_count += 1

    r_score = matched_count / float(n_events)
    return EvaluationResult(
        task_type="TRAKE",
        query_id=query_id,
        r_score=round(r_score, 4),
        is_hit=matched_count == n_events,
        status="COMPLETED",
        details={"matched_events": matched_count, "total_events": n_events, "event_matches": event_matches},
    )


def calculate_r_at_k(ranked_r_scores: Sequence[float], k_values: Sequence[int] = (1, 5, 20, 50, 100)) -> dict[int, float]:
    """Calculate R@k = max(R-score in top k predictions) for each k in k_values."""
    out: dict[int, float] = {}
    for k in k_values:
        subset = ranked_r_scores[:k]
        out[k] = max(subset) if subset else 0.0
    return out


def calculate_final_score(r_at_k_map: dict[int, float]) -> float:
    """Calculate official Final Score = mean(R@1, R@5, R@20, R@50, R@100)."""
    k_targets = (1, 5, 20, 50, 100)
    scores = [r_at_k_map.get(k, 0.0) for k in k_targets]
    return round(sum(scores) / len(scores), 4)
