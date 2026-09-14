from __future__ import annotations

import pytest

from wp03.contracts import SearchCandidate
from wp03.fusion import RankedHit, diversify_visual_candidates, fuse_rrf


def hit(video_id: str, frame_id: int, rank: int, similarity: float) -> RankedHit:
    return RankedHit(
        vector_id=rank,
        video_id=video_id,
        frame_id=frame_id,
        timestamp_ms=frame_id * 100,
        keyframe_path=f"keyframes/{video_id}/{frame_id:06d}.jpg",
        rank=rank,
        similarity=similarity,
    )


def test_rrf_adds_rank_contributions_without_combining_similarity() -> None:
    result = fuse_rrf(
        {"beit3": [hit("L21_V001", 42, 1, 0.9)], "bge_vl": [hit("L21_V001", 42, 3, 0.1)]},
        query_id="q",
        event_index=4,
        preprocess_run_id="prep",
        limit=1,
    )

    assert result.candidates[0].score == pytest.approx(1 / 61 + 1 / 63)
    assert result.candidates[0].model_scores == {"beit3": 0.9, "bge_vl": 0.1}
    assert result.candidates[0].event_index == 4


def test_diversity_collapses_nearby_frames_then_round_robins_videos() -> None:
    candidates = (
        SearchCandidate.visual_rrf(query_id="q", event_index=None, preprocess_run_id="prep", video_id="A", frame_id=1,
                                   timestamp_ms=1_000, rank=1, rrf_score=0.9, model_scores={}, model_ranks={}, keyframe_path="a1.jpg"),
        SearchCandidate.visual_rrf(query_id="q", event_index=None, preprocess_run_id="prep", video_id="A", frame_id=2,
                                   timestamp_ms=1_500, rank=2, rrf_score=0.8, model_scores={}, model_ranks={}, keyframe_path="a2.jpg"),
        SearchCandidate.visual_rrf(query_id="q", event_index=None, preprocess_run_id="prep", video_id="B", frame_id=3,
                                   timestamp_ms=5_000, rank=3, rrf_score=0.7, model_scores={}, model_ranks={}, keyframe_path="b3.jpg"),
    )

    result = diversify_visual_candidates(candidates, limit=2, dedup_window_ms=1_000)

    assert [(candidate.video_id, candidate.frame_id, candidate.rank) for candidate in result] == [("A", 1, 1), ("B", 3, 2)]
