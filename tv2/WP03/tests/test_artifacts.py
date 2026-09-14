from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wp03.artifacts import EMBEDDING_MAP_SCHEMA, validate_embedding_array, write_embedding_map_parquet
from wp03.contracts import ContractError, EmbeddingMapRecord
from wp03.mapping import load_search_mapping


@pytest.mark.parametrize(
    "array",
    [np.array([[np.nan, 0]], np.float32), np.array([[0, 0]], np.float32)],
)
def test_embedding_array_rejects_non_searchable_vectors(array: np.ndarray) -> None:
    with pytest.raises(ContractError):
        validate_embedding_array(array, expected_rows=1, expected_dim=2)


def test_parquet_writer_uses_the_declared_embedding_map_schema(tmp_path: Path) -> None:
    record = EmbeddingMapRecord(
        schema_version="1.0.0",
        preprocess_run_id="prep-1",
        model_name="beit3",
        model_version="rev-1",
        vector_id=0,
        video_id="L21_V001",
        frame_id=42,
        keyframe_seq=7,
        timestamp_ms=12_345,
        shot_id="L21_V001-shot-0007",
        embedding_dim=2,
        vector_dtype="float32",
        l2_normalized=True,
        keyframe_path="keyframes/L21_V001/000042.jpg",
        created_at_utc="2026-08-06T00:00:00Z",
    )

    write_embedding_map_parquet(tmp_path / "map.parquet", [record])

    assert pq.read_schema(tmp_path / "map.parquet") == EMBEDDING_MAP_SCHEMA


def test_search_mapping_accepts_legacy_map_without_shot_id(tmp_path: Path) -> None:
    path = tmp_path / "legacy.parquet"
    pq.write_table(
        pa.table(
            {
                "video_id": ["L21_V001"],
                "frame_id": [42],
                "timestamp_ms": [12_345],
                "keyframe_path": ["keyframes/L21_V001/000042.jpg"],
            }
        ),
        path,
    )

    mapping = load_search_mapping(path)

    assert mapping.has_shot_ids is False
    assert mapping[0]["frame_id"] == 42
    assert mapping.shot_neighbor_ids(0, 4) == ()
