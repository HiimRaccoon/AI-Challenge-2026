from wp04.contracts import ASRSegment, FrameRecord, OCRDetection
from wp04.metadata import build_metadata, collection_metadata
from wp04.temporal import LocalTemporalResolver


def test_video_metadata_projects_to_midpoint_frame_without_observed_text():
    frames = [
        FrameRecord("tv1", "v", 10, 1, 1000),
        FrameRecord("tv1", "v", 50, 2, 5000),
        FrameRecord("tv1", "v", 90, 3, 9000),
    ]
    record = build_metadata("tv1", "v", frames, {"duration_ms": 9000, "width": 1920}, [], [], LocalTemporalResolver(frames, []))
    assert record.frame_id == 50
    assert record.fields == {"duration_ms": 9000, "width": 1920}


def test_metadata_uses_observed_ocr_evidence_and_never_invents_caption():
    frame = FrameRecord("tv1", "v", 42, 1, 1400)
    ocr = OCRDetection("tv1", "v", 42, 1400, "BÁNH MÌ", "bánh mì", (0.1, 0.2, 0.3, 0.4), 0.9, "m", "v", "ocr:v:42:0")
    record = build_metadata("tv1", "v", [frame], {}, [ocr], [], LocalTemporalResolver([frame], []))
    assert record.frame_id == 42
    assert record.fields["ocr_text"] == ["BÁNH MÌ"]
    assert "caption" not in record.fields
    assert record.evidence_refs == ("ocr:v:42:0",)


def test_metadata_uses_tv1_frame_span_midpoint_when_duration_is_unknown():
    frames = [
        FrameRecord("tv1", "v", 10, 1, 1000),
        FrameRecord("tv1", "v", 50, 2, 5000),
        FrameRecord("tv1", "v", 90, 3, 9000),
    ]
    record = build_metadata("tv1", "v", frames, {}, [], [], LocalTemporalResolver(frames, []))
    assert record.frame_id == 50


def test_l_prefix_is_navigation_metadata_not_a_topic_label():
    frames = [FrameRecord("tv1", "L26_V003", 10, 1, 1000)]

    record = build_metadata(
        "tv1", "L26_V003", frames, {}, [], [], LocalTemporalResolver(frames, []),
    )

    assert record.fields["collection_id"] == "L26"
    assert record.fields["collection_label"] == "Videos_L26"
    assert "topic" not in record.fields
    assert collection_metadata("unstructured-video") == {}
