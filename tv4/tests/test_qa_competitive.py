from __future__ import annotations

from dataclasses import replace

from tv4.contracts import (
    AsrEvidence,
    CanonicalFrameReference,
    EvidencePack,
    OcrEvidence,
    SearchCandidate,
)
from tv4.qa_frame_selector import answer_support_span, select_answer_frames
from tv4.qa_reranker import (
    extract_typed_text_answer,
    qa_evidence_window_ms,
    rank_answerable_evidence,
    select_answerable_shortlist,
)


def _candidate(frame_id: int, timestamp_ms: int, score: float, rank: int) -> SearchCandidate:
    return SearchCandidate(
        query_id="qa-competitive",
        video_id="V1",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        source="fusion",
        rank=rank,
        score=score,
    )


def _pack(candidate: SearchCandidate, *, query_text=None, asr_texts=(), frames=()) -> EvidencePack:
    return EvidencePack(
        query_id=candidate.query_id,
        video_id=candidate.video_id,
        frame_id=candidate.frame_id,
        timestamp_ms=candidate.timestamp_ms,
        keyframe_path=None,
        query_text=query_text,
        selected_frames=frames,
        asr_texts=asr_texts,
    )


def test_answerability_promotes_answer_bearing_asr_window_over_visual_only_rank():
    visual = _candidate(10, 1_000, 1.0, 1)
    spoken = _candidate(20, 2_000, 0.72, 2)
    frames = (CanonicalFrameReference("V1", 10, 1_000, "10.jpg"),)

    ranked = rank_answerable_evidence(
        (visual, spoken),
        (
            _pack(visual, frames=frames),
            _pack(spoken, asr_texts=("Giá là hai trăm nghìn",), frames=frames),
        ),
        "Anh ấy nói giá bao nhiêu?",
    )

    assert ranked[0].candidate.frame_id == 20
    assert ranked[0].candidate.rank == 1
    assert ranked[0].answerability.modality_match == 1.0
    assert ranked[0].candidate.provenance["qa_retrieval_rank"] == 2


def test_answerability_promotes_semantically_matching_food_answer_over_higher_visual_rank():
    description = (
        "Tô cháo đã hoàn chỉnh, cạnh chén đen có topping màu cam dạng sợi, "
        "xung quanh có rau và hành."
    )
    generic = _candidate(10, 1_000, 1.0, 1)
    unrelated_seafood = _candidate(20, 2_000, 0.94, 2)
    salmon_porridge = _candidate(30, 3_000, 0.82, 7)
    frames = (CanonicalFrameReference("V1", 10, 1_000, "10.jpg"),)

    ranked = rank_answerable_evidence(
        (generic, unrelated_seafood, salmon_porridge),
        (
            _pack(
                generic, query_text=description,
                asr_texts=("Bây giờ chúng ta cho gia vị vào nồi",), frames=frames,
            ),
            _pack(
                unrelated_seafood, query_text=description,
                asr_texts=("Mực và tôm được xào với bơ",), frames=frames,
            ),
            _pack(
                salmon_porridge, query_text=description,
                asr_texts=("Tô cháo này dùng chà bông cá hồi",), frames=frames,
            ),
        ),
        "Topping trong video là thịt của con gì?",
    )

    assert ranked[0].candidate.frame_id == 30
    assert ranked[0].candidate.provenance["qa_retrieval_rank"] == 7
    assert ranked[0].answerability.semantic_answer_cue == 0.85
    assert ranked[1].candidate.frame_id == 20


def test_specific_animal_answer_outranks_generic_class_at_similar_retrieval_score():
    generic = _candidate(10, 1_000, 1.0, 1)
    specific = _candidate(20, 2_000, 0.97, 2)
    description = "Cô gái đặt các con X là nguyên liệu lên đĩa trắng"

    ranked = rank_answerable_evidence(
        (generic, specific),
        (
            _pack(generic, query_text=description, asr_texts=("Món cá hôm nay",)),
            _pack(specific, query_text=description, asr_texts=("Một con nghêu là nguyên liệu",)),
        ),
        "X là con gì?",
    )

    assert ranked[0].candidate.frame_id == 20


def test_open_qa_context_policy_depends_on_retrieval_position_not_question_template():
    assert qa_evidence_window_ms(0) == 30_000
    assert qa_evidence_window_ms(7) == 30_000
    assert qa_evidence_window_ms(15) == 30_000
    assert qa_evidence_window_ms(16) == 3_000
    assert qa_evidence_window_ms(29) == 3_000


def test_open_qa_shortlist_keeps_top_retrieval_videos_for_unseen_answer_types():
    candidates = tuple(
        replace(
            _candidate(frame_id=index, timestamp_ms=index * 1_000, score=1.0 / index, rank=index),
            video_id=f"V{index}",
        )
        for index in range(1, 7)
    )
    evidence = tuple(_pack(candidate, asr_texts=(f"novel evidence {candidate.frame_id}",)) for candidate in candidates)
    reranked = rank_answerable_evidence(candidates, evidence, "Một câu hỏi hoàn toàn mới?")
    # Simulate an evidence scorer that puts retrieval ranks 4-6 first.
    reordered = [*reranked[3:], *reranked[:3]]

    selected = select_answerable_shortlist(reordered, candidates, limit=5)
    selected_frames = {item.candidate.frame_id for item in selected}

    assert {1, 2, 3} <= selected_frames
    assert len(selected) == 5


def test_animal_answer_detection_keeps_vietnamese_accents_to_avoid_false_matches():
    question = "Topping là thịt của con gì?"

    assert extract_typed_text_answer(question, "Chà bông cá hồi rắc trên tô cháo") == "Cá hồi"
    assert extract_typed_text_answer(question, "Đây là món của đầu bếp để ăn sáng") == ""
    assert extract_typed_text_answer(question, "Một con nghêu chứa nhiều chất sắt") == "Nghêu"


def test_answer_frame_selector_returns_only_tv1_frames_and_localizes_ocr_answer():
    evidence = EvidencePack(
        query_id="qa-frame",
        video_id="V1",
        frame_id=100,
        timestamp_ms=4_000,
        keyframe_path="100.jpg",
        selected_frames=(
            CanonicalFrameReference("V1", 100, 4_000, "100.jpg"),
            CanonicalFrameReference("V1", 110, 4_400, "110.jpg"),
        ),
        ocr_evidence=(
            OcrEvidence(
                detection_id="ocr-110",
                video_id="V1",
                frame_id=110,
                timestamp_ms=4_400,
                raw_text="GIANG LY",
                normalized_text="giang ly",
                bbox_xyxy_norm=(0.1, 0.1, 0.8, 0.3),
            ),
        ),
    )

    selected = select_answer_frames("Giang Ly", evidence, verified=True)

    assert selected[0].frame_id == 110
    assert {item.frame_id for item in selected} <= {100, 110}
    assert "ocr" in selected[0].support_modalities
    assert answer_support_span("Giang Ly", evidence) == (4_400, 4_400)


def test_answer_support_span_uses_matching_asr_boundaries():
    evidence = EvidencePack(
        query_id="qa-asr",
        video_id="V1",
        frame_id=10,
        timestamp_ms=1_000,
        keyframe_path=None,
        asr_evidence=(
            AsrEvidence(
                segment_id="s1", video_id="V1", start_ms=900, end_ms=1_500,
                text="Giá là hai trăm nghìn",
            ),
        ),
    )
    assert answer_support_span("hai trăm nghìn", evidence) == (900, 1_500)


def test_answer_support_span_uses_timed_neighbor_asr_context():
    evidence = EvidencePack(
        query_id="qa-asr-context",
        video_id="V1",
        frame_id=10,
        timestamp_ms=4_000,
        keyframe_path=None,
        asr_evidence=(
            AsrEvidence(
                segment_id="s2", video_id="V1", start_ms=3_500, end_ms=4_500,
                text="Tô cháo đã hoàn chỉnh",
                context=({
                    "start_ms": 4_600,
                    "end_ms": 5_400,
                    "text": "Topping là chà bông cá hồi",
                },),
            ),
        ),
    )

    assert answer_support_span("Cá hồi", evidence) == (4_600, 5_400)
