from __future__ import annotations

import pytest

from wp08.contracts import CandidateId, FeedbackEvent, FeedbackValidationError


def test_feedback_event_rejects_whitespace_only_text() -> None:
    with pytest.raises(FeedbackValidationError, match="non-whitespace"):
        FeedbackEvent.create(candidate_id=CandidateId("L21_V001", 42), feedback_text="   ")


def test_feedback_event_rejects_more_than_300_raw_characters() -> None:
    with pytest.raises(FeedbackValidationError, match="300"):
        FeedbackEvent.create(candidate_id=CandidateId("L21_V001", 42), feedback_text="x" * 301)
