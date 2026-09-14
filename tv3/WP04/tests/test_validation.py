from wp04.contracts import FrameRecord, MetadataRecord, OCRDetection, WP04RunIdentity
from wp04.storage import ArtifactStore
from wp04.validation import validate_records, validate_run


def test_validation_resolves_every_evidence_ref_for_valid_records():
    frame = FrameRecord("tv1", "v", 42, 1, 1400)
    ocr = OCRDetection("tv1", "v", 42, 1400, "BÁNH MÌ", "bánh mì", (0.1, 0.2, 0.3, 0.4), 0.9, "m", "v", "ocr:v:42:0")
    metadata = MetadataRecord("tv1", "v", 42, 1400, {"ocr_text": ["BÁNH MÌ"]}, (ocr.evidence_id,), "metadata:v:42")
    assert validate_records([frame], [ocr], [], [], [metadata]).errors == []


def test_validation_reports_cross_video_frame_and_unknown_evidence():
    frame = FrameRecord("tv1", "v1", 42, 1, 1400)
    ocr = OCRDetection("tv1", "v2", 42, 1400, "x", "x", (0.1, 0.2, 0.3, 0.4), 0.9, "m", "v", "ocr:v2:42:0")
    metadata = MetadataRecord("tv1", "v1", 42, 1400, {}, ("missing",), "metadata:v1:42")
    codes = [issue.code for issue in validate_records([frame], [ocr], [], [], [metadata]).errors]
    assert codes == ["OCR_FRAME_NOT_IN_TV1", "UNKNOWN_EVIDENCE_REF"]
    assert validate_records([frame], [ocr], [], [], [metadata]).to_dict()["valid"] is False


def test_validate_run_reads_shards_and_writes_report(tmp_path):
    frame = FrameRecord("tv1", "v", 42, 1, 1400)
    ocr = OCRDetection("tv1", "v", 42, 1400, "x", "x", (0.1, 0.2, 0.3, 0.4), 0.9, "m", "v", "ocr:v:42:0")
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))
    store.commit_video("ocr", "v", [ocr], "fp")
    report = validate_run(tmp_path, store.identity, [frame])
    assert report.is_valid
    assert (tmp_path / "wp04" / "wp04" / "reports" / "wp04-validation.json").exists()
