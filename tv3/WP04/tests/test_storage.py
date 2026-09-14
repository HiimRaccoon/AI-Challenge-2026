from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from wp04.contracts import OCRDetection, WP04RunIdentity
from wp04.storage import ArtifactStore, SimulatedCrash
from wp04.validation import validate_run
from wp04.contracts import FrameRecord


def identity() -> WP04RunIdentity:
    return WP04RunIdentity("tv1-a", "wp04-a", "inputs-a", "config-a")


def ocr(video_id: str = "L01_V001") -> OCRDetection:
    return OCRDetection(
        "tv1-a", video_id, 42, 1400, "BÁNH MÌ", "bánh mì", (0.1, 0.2, 0.3, 0.4),
        0.9, "craft-parseq-vn", "v1", f"ocr:{video_id}:42:0",
    )


def test_ready_is_not_written_before_durable_video_shard(tmp_path: Path):
    store = ArtifactStore(tmp_path, identity())
    with pytest.raises(SimulatedCrash):
        store.commit_video("ocr", "L01_V001", [ocr()], "fp", crash_before_status=True)
    assert store.status_for("L01_V001", "ocr") is None
    assert store.read_records("ocr", "L01_V001") == [ocr().to_dict()]


def test_completed_requires_matching_fingerprint_and_durable_shard(tmp_path: Path):
    store = ArtifactStore(tmp_path, identity())
    store.commit_video("ocr", "L01_V001", [ocr()], "original")
    assert store.completed("L01_V001", "ocr", "original")
    assert not store.completed("L01_V001", "ocr", "changed-input")


def test_store_rejects_frame_from_another_video(tmp_path: Path):
    store = ArtifactStore(tmp_path, identity())
    with pytest.raises(ValueError, match="outside its video"):
        store.validate_ocr([ocr("L02_V001")], {("L01_V001", 42)})


def test_ready_allows_a_durable_empty_detection_shard(tmp_path: Path):
    store = ArtifactStore(tmp_path, identity())
    store.commit_video("ocr", "L01_V001", [], "fp")
    assert store.status_for("L01_V001", "ocr").state == "ready"
    assert store.read_records("ocr", "L01_V001") == []


def test_promote_compacts_only_a_validated_artifact_set(tmp_path: Path):
    store = ArtifactStore(tmp_path, identity())
    store.commit_video("ocr", "L01_V001", [ocr()], "fp")
    frame = FrameRecord("tv1-a", "L01_V001", 42, 1, 1400)
    assert validate_run(tmp_path, identity(), [frame]).is_valid
    promoted = store.promote(["ocr"])
    assert promoted["ocr"] == tmp_path / "ocr" / "ocr.parquet"
    assert promoted["ocr"].exists()


def test_parallel_status_updates_preserve_every_video(tmp_path: Path):
    store = ArtifactStore(tmp_path, identity())
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: store.append_status(__import__("wp04.status", fromlist=["ModalityStatus"]).ModalityStatus.ready(f"v{index}", "ocr", "fp")), range(8)))
    assert {status.video_id for status in store._all_statuses()} == {f"v{index}" for index in range(8)}
