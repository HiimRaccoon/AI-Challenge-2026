from __future__ import annotations

import pytest

from wp03.contracts import ContractError, SearchCandidate
from wp03.feedback_pool import EmbeddingReference, FeedbackPoolCandidate, FeedbackPoolSnapshot


def candidate() -> SearchCandidate:
    return SearchCandidate.visual_rrf(
        query_id="query-1",
        event_index=None,
        preprocess_run_id="prep-1",
        video_id="L21_V001",
        frame_id=42,
        timestamp_ms=1_000,
        rank=1,
        rrf_score=0.1,
        model_scores={},
        model_ranks={},
        keyframe_path="keyframes/L21_V001/000042.jpg",
    )


def ref(model_key: str) -> EmbeddingReference:
    return EmbeddingReference(model_key=model_key, vector_id=1, wp03_run_id="run-1", mapping_sha256="a" * 64)


def test_feedback_pool_requires_one_reference_per_requested_model() -> None:
    item = FeedbackPoolCandidate(candidate=candidate(), fused_rank=1, embedding_refs={"beit3": ref("beit3")})

    with pytest.raises(ContractError, match="embedding references"):
        FeedbackPoolSnapshot.create(
            query_id="query-1",
            wp03_run_id="run-1",
            models=("beit3", "bge_vl"),
            pool_size=500,
            rrf_k=60,
            candidates=(item,),
        )
