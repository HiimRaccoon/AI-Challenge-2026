from tv4.kis_pipeline import _question_prefers_asr
from tv4.qa_dense_reader import DenseOcrReader, ProvenJpegFrame
from tv4.qa_query_plan import build_qa_query_plan
from tv4.qa_reranker import preferred_modalities
from tv4.contracts import EvidencePack, OcrEvidence
from tv4.wp11_vqa import _direct_ocr_answer, _direct_ocr_anchor
from tv4.external_assist import ExternalQueryAssist


def test_scale_display_plan_preserves_device_relation():
    plan = build_qa_query_plan(
        "Một con cá được đặt lên cân điện tử rồi người khác cầm cá.",
        "Con số hiển thị cuối cùng trên cân là bao nhiêu?",
    )
    assert plan.answer_source == "device_display"
    assert plan.device == "weighing scale"
    assert plan.temporal_rule == "last_before_scene_change"
    assert plan.relation_query == "close-up of the display attached to the weighing scale"
    assert "price" not in plan.relation_query


def test_unrelated_price_board_and_generic_machine_screen_are_not_scale_expansions():
    price = build_qa_query_plan("Một bảng giá trong chợ", "Giá cuối cùng là bao nhiêu?")
    machine = build_qa_query_plan("Một máy đang hoạt động", "Số trên màn hình là bao nhiêu?")
    assert not price.requires_dense_read
    assert not machine.requires_dense_read
    assert price.relation_query is None
    assert machine.relation_query is None


def test_visual_display_question_does_not_get_asr_first():
    question = "Con số hiển thị cuối cùng trên cân là bao nhiêu?"
    assert _question_prefers_asr(question, "cá được đặt lên cân điện tử") is False
    assert preferred_modalities(question)[0] == "ocr"
    assert _question_prefers_asr("Ai nói câu này?") is True


def test_vqa_query_set_reserves_device_relation_without_generic_screens():
    assist = ExternalQueryAssist(enabled=False, base_url="", model="")
    description = "a fish is placed on a scale, then another fish is held by its tail"
    queries = assist.vqa_visual_queries(description, "What is the final number on the scale display?")
    assert queries[0] == description
    assert queries[-1] == "close-up of the display attached to the weighing scale"
    assert not any("price board" in query or "machine screen" in query for query in queries)
    assert len(queries) == 2


def test_dense_reader_selects_last_stable_value_not_earlier_value():
    frames = [ProvenJpegFrame("L21_V007", 8840 + index, 294_000 + index * 33, b"jpg") for index in range(6)]

    def transport(url, payload, timeout):
        assert url.endswith("/ocr/batch")
        rows = []
        for index in range(6):
            value = "38.25" if index < 3 else "37.05"
            rows.append([
                {"text": "06:35:24", "confidence": 1.0, "bbox_xyxy_norm": [0.8, 0.1, 0.9, 0.2]},
                {"text": value, "confidence": 0.99, "bbox_xyxy_norm": [0.1, 0.1, 0.3, 0.2]},
            ])
        return {"detections": rows}

    reader = DenseOcrReader(enabled=True, transport=transport, min_stable_frames=2, max_batch_frames=8)
    plan = build_qa_query_plan("cá trên cân điện tử", "Số hiển thị cuối cùng trên cân là bao nhiêu?")
    result = reader.read(frames, plan)
    assert result.available is True
    assert result.selected is not None
    assert result.selected.normalized_value == "37.05"
    assert result.selected.frame_id == 8844
    assert result.support_frame_ids == (8843, 8844, 8845)


def test_dense_reader_rejects_one_frame_numeric_flash():
    frames = [ProvenJpegFrame("v", index, index * 33, b"jpg") for index in range(3)]

    def transport(url, payload, timeout):
        return {"detections": [[], [{"text": "99.9", "confidence": 0.99, "bbox_xyxy_norm": [0, 0, 1, 1]}], []]}

    reader = DenseOcrReader(enabled=True, transport=transport, min_stable_frames=2, max_batch_frames=3)
    plan = build_qa_query_plan("vật trên cân", "Số cuối cùng trên cân là bao nhiêu?")
    result = reader.read(frames, plan)
    assert result.available is False
    assert result.reason == "no_stable_numeric_reading"


def test_chart_question_routes_to_visual_chart_reading():
    plan = build_qa_query_plan(
        "Biểu đồ so sánh tốc độ sinh trưởng của các loài cây ven biển.",
        "Loài cây II tốt nhất ở độ mặn bao nhiêu phần nghìn?",
    )

    assert plan.answer_source == "chart_reading"
    assert plan.operation == "read_chart_relation"
    assert plan.requires_visual_read is True
    assert plan.requires_dense_read is False
    assert plan.prefer_video_consensus is True
    assert "chart" in (plan.relation_query or "")


def test_missing_visible_number_routes_to_bounded_set_difference():
    plan = build_qa_query_plan(
        "Các cột biểu diễn được dán số.",
        "Trong 16 giây đầu tiên, số nào không được nhìn thấy trong các số từ 1-8?",
    )

    assert plan.answer_source == "temporal_set_difference"
    assert plan.operation == "set_difference"
    assert plan.requires_multi_frame is True
    assert plan.interval_start_ms == 0
    assert plan.interval_end_ms == 16_000
    assert plan.candidate_values == tuple(str(value) for value in range(1, 9))


def test_certified_stable_dense_ocr_is_a_direct_grounded_answer():
    ocr = OcrEvidence(
        detection_id="dense", video_id="L21_V007", frame_id=8845,
        timestamp_ms=294833, raw_text="37.05", normalized_text="37.05",
        bbox_xyxy_norm=(0.37, 0.47, 0.47, 0.56), confidence=0.97,
        provenance={"source": "wp09_certified_dense_decode"},
    )
    evidence = EvidencePack(
        query_id="q", video_id="L21_V007", frame_id=8845,
        timestamp_ms=294833, keyframe_path=None,
        query_text="cá được đặt lên cân điện tử",
        question="Con số hiển thị cuối cùng trên cân là bao nhiêu?",
        ocr_evidence=(ocr,), ocr_texts=("37.05",),
        availability={"ocr": "available"},
        provenance={
            "dense_ocr": {
                "available": True,
                "selected": {"normalized_value": "37.05", "frame_id": 8845},
                "support_frame_ids": [8844, 8845],
            },
        },
    )
    assert _direct_ocr_answer(evidence.question or "", evidence) == "37.05"
    assert _direct_ocr_anchor("37.05", evidence) == ocr
