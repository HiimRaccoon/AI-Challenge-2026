from wp04.contracts import OCRDetection
from wp04.evidence import EvidenceResolver, assign_ocr_evidence_ids


def test_ocr_evidence_ids_are_deterministic_after_box_and_text_sorting():
    records = [
        OCRDetection("tv1", "v", 1, 10, "z", "z", (0.5, 0.2, 0.6, 0.3), 0.8, "m", "v", "pending"),
        OCRDetection("tv1", "v", 1, 10, "a", "a", (0.1, 0.2, 0.3, 0.4), 0.9, "m", "v", "pending"),
    ]
    assigned = assign_ocr_evidence_ids(records)
    assert [item.evidence_id for item in assigned] == ["ocr:v:1:1", "ocr:v:1:0"]
    assert EvidenceResolver([*assigned]).resolve_evidence("ocr:v:1:0").raw_text == "a"
