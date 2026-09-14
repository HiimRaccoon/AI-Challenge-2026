from __future__ import annotations

import numpy as np

from wp03.feedback_pool import EmbeddingReference, FeedbackPoolCandidate, FeedbackPoolSnapshot
from wp03.contracts import SearchCandidate
from wp08.wp03_adapter import FourModelFeedbackRanker


MODELS = ("beit3", "bge_vl", "metaclip2", "perception")


def candidate(video_id: str, frame_id: int, rank: int) -> FeedbackPoolCandidate:
    raw = SearchCandidate.visual_rrf(query_id="q", event_index=None, preprocess_run_id="p", video_id=video_id, frame_id=frame_id, timestamp_ms=frame_id, rank=rank, rrf_score=1.0, model_scores={}, model_ranks={}, keyframe_path="x.jpg")
    refs = {model: EmbeddingReference(model, rank, "run", "a" * 64) for model in MODELS}
    return FeedbackPoolCandidate(raw, rank, refs)


def test_four_model_ranker_fuses_each_model_then_late_rrf() -> None:
    first, second = candidate("A", 1, 1), candidate("B", 2, 2)
    snapshot = FeedbackPoolSnapshot.create(query_id="q", wp03_run_id="run", models=MODELS, pool_size=500, rrf_k=60, candidates=(first, second))
    vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    ranker = FourModelFeedbackRanker(snapshot, {model: lambda _: np.array([[1.0, 0.0]]) for model in MODELS}, lambda ref: vectors[ref.vector_id])
    assert ranker("query", first.candidate.video_id, first.candidate.frame_id)[0].video_id == "A"
