import pytest

from wp04.contracts import (
    ASRSegment,
    AudioRecord,
    ContractError,
    MetadataRecord,
    OCRDetection,
    ObjectDetection,
    SearchCandidate,
    SearchRequest,
    WP04RunIdentity,
)


def test_search_candidate_mirrors_pipeline_fields_and_allows_rank_over_100():
    candidate = SearchCandidate(query_id="q", video_id="L01_V001", frame_id=42,
        timestamp_ms=1400, source="ocr", rank=101, preprocess_run_id="tv1-run")
    assert set(candidate.to_dict()) == {
        "schema_version", "query_id", "event_index", "video_id", "frame_id",
        "timestamp_ms", "source", "raw_score", "score", "rank", "model_scores",
        "model_ranks", "matched_filters", "evidence_refs", "confidence",
        "preprocess_run_id", "created_at_utc",
    }


def test_search_request_preserves_router_fields_and_rejects_limit_over_100():
    request = SearchRequest(query_id="q", task="TRAKE", query_text=None,
        question="what", events=("first",), filters={"kind": "x"}, limit=101,
        language="vi", session_id="s")
    with pytest.raises(ContractError):
        request.validate()


def test_modality_records_keep_original_identity_and_stable_evidence_ids():
    ocr = OCRDetection(
        preprocess_run_id="tv1-a", video_id="L01_V001", frame_id=42,
        timestamp_ms=1400, raw_text="BÁNH MÌ", normalized_text="bánh mì",
        bbox_xyxy_norm=(0.1, 0.2, 0.3, 0.4), confidence=0.9,
        model_name="craft-parseq-vn", model_version="v1", evidence_id="ocr:L01_V001:42:0",
    )
    segment = ASRSegment(
        preprocess_run_id="tv1-a", video_id="L01_V001", segment_id="asr:L01_V001:0",
        start_ms=1000, end_ms=1600, raw_text="xin chào", normalized_text="xin chào",
        confidence=0.8, model_name="chunkformer-ctc-large-vie", model_version="v1",
    )
    detected = ObjectDetection(
        preprocess_run_id="tv1-a", video_id="L01_V001", frame_id=42,
        timestamp_ms=1400, label="person", bbox_xyxy_norm=(0.1, 0.2, 0.3, 0.4),
        confidence=0.7, model_name="rf-detr", model_version="v1", evidence_id="object:L01_V001:42:0",
    )
    metadata = MetadataRecord(
        preprocess_run_id="tv1-a", video_id="L01_V001", frame_id=42,
        timestamp_ms=1400, fields={"ocr_text": ["BÁNH MÌ"]},
        evidence_refs=(ocr.evidence_id,), record_id="metadata:L01_V001:42",
    )
    assert ocr.to_dict()["frame_id"] == 42
    assert segment.to_dict()["segment_id"] == "asr:L01_V001:0"
    assert detected.to_dict()["label"] == "person"
    assert metadata.to_dict()["evidence_refs"] == ["ocr:L01_V001:42:0"]


def test_run_identity_and_audio_record_reject_invalid_time_ranges():
    identity = WP04RunIdentity("tv1-a", "wp04-a", "inputs", "config")
    assert identity.wp04_artifact_set_id == "wp04-a"
    with pytest.raises(ContractError):
        AudioRecord("tv1-a", "L01_V001", "audio.wav", "abc", True, -1)
    with pytest.raises(ContractError):
        ASRSegment("tv1-a", "L01_V001", "s", 3, 2, "", "", 0.0, "m", "v")


def test_frame_record_carries_keyframe_path_and_checksum_for_real_inference():
    frame = __import__("wp04.contracts", fromlist=["FrameRecord"]).FrameRecord(
        "tv1", "L01_V001", 42, 1, 1400, "keyframes/L01_V001/000042.jpg", "sha256-a",
    )
    assert frame.keyframe_path.endswith("000042.jpg")
    assert frame.keyframe_sha256 == "sha256-a"
