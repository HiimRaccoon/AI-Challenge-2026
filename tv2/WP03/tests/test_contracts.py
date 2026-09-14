from __future__ import annotations

import re

import pytest

from wp03.contracts import ContractError, FrameRecord, SearchCandidate, SearchRequest, SearchResponse


VALID_FRAME = {
    "preprocess_run_id": "prep-1",
    "video_id": "L21_V001",
    "frame_id": 42,
    "keyframe_seq": 7,
    "timestamp_ms": 12_345,
    "pts": 296_280,
    "time_base": "1/24000",
    "decode_index": 296,
    "shot_id": "L21_V001_S0007",
    "keyframe_path": "keyframes/L21_V001/000042.jpg",
}


def test_visual_candidate_serializes_all_pipeline_fields() -> None:
    candidate = SearchCandidate.visual_rrf(
        query_id="q-1",
        event_index=None,
        preprocess_run_id="prep-1",
        video_id="L21_V001",
        frame_id=42,
        timestamp_ms=12_345,
        rank=1,
        rrf_score=1 / 62,
        model_scores={"beit3": 0.41},
        model_ranks={"beit3": 2},
        keyframe_path="keyframes/L21_V001/000042.jpg",
    )

    body = candidate.to_dict()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", body["created_at_utc"])
    body["created_at_utc"] = "<utc>"
    assert body == {
        "schema_version": "1.0.0",
        "query_id": "q-1",
        "event_index": None,
        "video_id": "L21_V001",
        "frame_id": 42,
        "timestamp_ms": 12_345,
        "source": "visual",
        "raw_score": None,
        "score": 1 / 62,
        "rank": 1,
        "model_scores": {"beit3": 0.41},
        "model_ranks": {"beit3": 2},
        "matched_filters": [],
        "evidence_refs": ["keyframe:keyframes/L21_V001/000042.jpg"],
        "confidence": None,
        "preprocess_run_id": "prep-1",
        "created_at_utc": "<utc>",
    }


def test_frame_record_rejects_negative_frame_id() -> None:
    with pytest.raises(ContractError, match="frame_id"):
        FrameRecord.from_dict({**VALID_FRAME, "frame_id": -1})


def test_search_response_exposes_candidate_limit_metadata() -> None:
    response = SearchResponse.create(
        query_id="q-1",
        wp03_run_id="wp03-1",
        preprocess_run_id="prep-1",
        requested_top_k=100,
        candidate_k_per_model=20,
        hard_candidate_cap=20,
        models_requested=("beit3",),
        models_used=("beit3",),
        candidates=(),
    )
    assert response.returned_count == 0
    assert response.hard_candidate_cap == 20


def test_pipeline_search_request_uses_question_when_query_text_is_missing() -> None:
    request = SearchRequest(
        query_id="q-1", task="VQA", query_text=None, question="What vehicle is visible?",
        events=(), filters={"video_id": "L21_V001"}, limit=100, language="en", session_id="s-1",
    )

    assert request.visual_query_text() == "What vehicle is visible?"


@pytest.mark.parametrize("limit", [0, 101])
def test_pipeline_search_request_rejects_limit_outside_top_100(limit: int) -> None:
    with pytest.raises(ContractError, match="limit"):
        SearchRequest(
            query_id="q-1", task="KIS", query_text="car", question=None,
            events=(), filters={}, limit=limit, language=None, session_id=None,
        )
