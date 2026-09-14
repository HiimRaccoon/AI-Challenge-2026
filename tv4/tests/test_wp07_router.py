from tv4.wp07_router import route_kis, route_qa, route_trake, split_trake_events
from tv4.kis_pipeline import _question_prefers_asr


def test_route_kis_always_includes_visual():
    decision = route_kis("Tìm cảnh một người đang mở laptop trong kho video.")
    assert "visual" in decision.branches
    assert decision.request.task == "KIS"


def test_route_kis_adds_ocr_hint():
    decision = route_kis("Tìm cảnh có dòng chữ khuyến mãi trên biển hiệu.")
    assert "ocr" in decision.branches


def test_route_qa_carries_question():
    decision = route_qa("video lễ trao giải", "có bao nhiêu người lên sân khấu?")
    assert decision.request.question == "có bao nhiêu người lên sân khấu?"
    assert decision.request.task == "VQA"
    assert decision.branches == ("visual", "ocr", "asr", "object", "metadata")


def test_location_questions_prefer_asr_candidates_for_vqa():
    assert _question_prefers_asr("cháy rừng xảy ra ở đâu?") is True
    assert _question_prefers_asr("what color is the car?") is False


def test_split_trake_events_numbered():
    text = "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
    events = split_trake_events(text)
    assert len(events) == 4
    assert events[0].startswith("giậm nhảy")


def test_route_trake_uses_explicit_events_when_given():
    decision = route_trake("bất kỳ", events=["a", "b", "c"])
    assert decision.request.events == ("a", "b", "c")
