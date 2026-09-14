from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from wp03.contracts import ContractError, FrameRecord
from wp03.corpus import load_corpus, resolve_keyframe
from tests.conftest import write_jsonl


def record(frame_id: int, keyframe_path: str) -> dict[str, object]:
    return {
        "preprocess_run_id": "prep-1",
        "video_id": "L21_V001",
        "frame_id": frame_id,
        "keyframe_seq": frame_id,
        "timestamp_ms": frame_id * 100,
        "pts": frame_id,
        "time_base": "1/24000",
        "decode_index": frame_id,
        "shot_id": "L21_V001_S0001",
        "keyframe_path": keyframe_path,
    }


def test_load_corpus_sorts_by_video_and_frame_id(data_root: Path) -> None:
    write_jsonl(
        data_root / "frames.jsonl",
        [record(42, "keyframes/L21_V001/000042.jpg"), record(3, "keyframes/L21_V001/000003.jpg")],
    )

    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)

    assert [(item.video_id, item.frame_id) for item in corpus] == [("L21_V001", 3), ("L21_V001", 42)]


def test_load_corpus_reads_the_same_records_from_parquet(data_root: Path) -> None:
    records = [record(42, "keyframes/L21_V001/000042.jpg"), record(3, "keyframes/L21_V001/000003.jpg")]
    write_jsonl(data_root / "frames.jsonl", records)
    pq.write_table(pa.Table.from_pylist(records), data_root / "frames.parquet")

    from_jsonl = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    from_parquet = load_corpus(data_root, PurePosixPath("frames.parquet"), None)

    assert from_parquet == from_jsonl


def test_load_corpus_rejects_duplicate_video_frame(data_root: Path) -> None:
    write_jsonl(
        data_root / "duplicate.jsonl",
        [record(3, "keyframes/L21_V001/000003.jpg"), record(3, "keyframes/L21_V001/000042.jpg")],
    )

    with pytest.raises(ContractError, match="duplicate"):
        load_corpus(data_root, PurePosixPath("duplicate.jsonl"), None)


@pytest.mark.parametrize("unsafe_path", ["../outside.jpg", "..\\outside.jpg", "C:\\x.jpg", "\\\\server\\x.jpg"])
def test_resolve_keyframe_rejects_unsafe_path(data_root: Path, unsafe_path: str) -> None:
    frame = FrameRecord.from_dict(record(3, unsafe_path))

    with pytest.raises(ContractError, match="keyframe_path"):
        resolve_keyframe(data_root, frame)
