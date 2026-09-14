from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest
from tv5.readiness import ReadinessConfig, validate_readiness

def _wp03(root: Path, **changes: object) -> Path:
    root.mkdir(); data={"preprocess_run_id":"run","videos":["a","b"],"vector_count":2,"selected_frames":2,"mapping_sha256":"a"*64,"model_version":"m","index_sha256":"b"*64}; data.update(changes)
    (root/"manifest.json").write_text(json.dumps(data)); return root

def test_all_readiness_statuses_and_input_bytes_unchanged(tmp_path: Path) -> None:
    root=_wp03(tmp_path/"ready"); before=(root/"manifest.json").read_bytes()
    assert validate_readiness(ReadinessConfig.wp03(root,expected_videos=["a","b"])).status=="READY"
    assert (root/"manifest.json").read_bytes()==before
    assert validate_readiness(ReadinessConfig.wp03(_wp03(tmp_path/"partial"),expected_videos=["a","b","c"])).status=="PARTIAL"
    base=tmp_path/"none"
    assert validate_readiness(ReadinessConfig("x","wp04",base,known_handover=True)).status=="HANDOVER PENDING"
    assert validate_readiness(ReadinessConfig("x","wp04",base,upstream_capability=True)).status=="CODE GAP"
    assert validate_readiness(ReadinessConfig("x","wp04",base)).status=="ACTUALLY MISSING"

@pytest.mark.parametrize("changes", [{"vector_count":3},{"mapping_sha256":""}])
def test_wp03_incompatible_index_or_linkage(tmp_path: Path, changes: dict[str, object]) -> None:
    assert validate_readiness(ReadinessConfig.wp03(_wp03(tmp_path/"bad",**changes),expected_videos=["a","b"])).status=="INCOMPATIBLE"

def test_digest_mismatch_and_wp04_modality_records(tmp_path: Path) -> None:
    root=_wp03(tmp_path/"digest"); assert validate_readiness(ReadinessConfig.wp03(root,expected_videos=["a","b"],expected_digest="0"*64)).status=="INCOMPATIBLE"
    wp04=tmp_path/"ocr"; wp04.mkdir(); (wp04/"manifest.json").write_text(json.dumps({"preprocess_run_id":"r","model_version":"m"})); (wp04/"records.json").write_text(json.dumps([{ "video_id":"v","frame_id":1,"timestamp_ms":2,"text":"x","bbox":[1,2,3,4]}])); (wp04/"index.bin").write_bytes(b"x")
    c=ReadinessConfig("OCR","wp04",wp04,modality="OCR",index_name="index.bin")
    assert validate_readiness(c).status=="READY"
    (wp04/"records.json").write_text(json.dumps([{ "video_id":"v","frame_id":1,"timestamp_ms":2,"text":"x"}]))
    assert validate_readiness(c).status=="INCOMPATIBLE"

@pytest.mark.parametrize("modality,row,bad", [
    ("OCR", {"video_id":"v","frame_id":1,"timestamp_ms":2,"text":"x","bbox":[1]}, {"video_id":"v","frame_id":1,"timestamp_ms":2,"text":"x"}),
    ("ASR", {"video_id":"v","start_ms":1,"end_ms":2,"transcript":"x","context":"x"}, {"video_id":"v","start_ms":3,"end_ms":2,"transcript":"x","context":"x"}),
    ("Object", {"video_id":"v","frame_id":1,"timestamp_ms":2,"label":"car","bbox":[1],"crop_ref":"x"}, {"video_id":"v","frame_id":1,"timestamp_ms":2,"label":"car","bbox":[1]}),
    ("Metadata", {"video_id":"v","evidence_value":"x"}, {"video_id":"v"}),
])
def test_all_wp04_modalities_valid_and_invalid(tmp_path: Path, modality: str, row: dict, bad: dict) -> None:
    root=tmp_path/modality; root.mkdir(); (root/"manifest.json").write_text(json.dumps({"preprocess_run_id":"r","model_version":"m"})); (root/"index.bin").write_bytes(b"x")
    c=ReadinessConfig(modality,"wp04",root,modality=modality,index_name="index.bin")
    (root/"records.json").write_text(json.dumps([row])); assert validate_readiness(c).status=="READY"
    (root/"records.json").write_text(json.dumps([bad])); assert validate_readiness(c).status=="INCOMPATIBLE"
