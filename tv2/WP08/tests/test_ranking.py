from __future__ import annotations

import numpy as np
import pytest

from wp08.contracts import CandidateId, FeedbackEvent, FeedbackValidationError
from wp08.ranking import fuse_embedding, late_rrf
from wp08.text import build_feedback_template, validate_token_budget


def test_template_preserves_original_query_and_raw_feedback() -> None:
    event = FeedbackEvent.create(candidate_id=CandidateId("L21_V001", 42), feedback_text="find it at night")
    assert build_feedback_template("a bus stop", (event,)) == "Original query: a bus stop\nRefinement 1: find it at night"


def test_token_budget_rejects_more_than_64_tokens() -> None:
    with pytest.raises(FeedbackValidationError, match="64"):
        validate_token_budget("x", token_counter=lambda _: 65)


def test_fusion_l2_normalizes_weighted_text_and_image() -> None:
    vector = fuse_embedding(np.array([3.0, 4.0]), np.array([0.0, 5.0]))
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_late_rrf_combines_four_complete_rank_lists() -> None:
    rankings = {
        "beit3": (CandidateId("A", 1), CandidateId("B", 2)),
        "bge_vl": (CandidateId("B", 2), CandidateId("A", 1)),
        "metaclip2": (CandidateId("A", 1), CandidateId("B", 2)),
        "perception": (CandidateId("A", 1), CandidateId("B", 2)),
    }
    assert late_rrf(rankings) == (CandidateId("A", 1), CandidateId("B", 2))
