"""Verified artifact writing for embeddings, mappings and manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import ContractError, EmbeddingMapRecord


EMBEDDING_MAP_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("preprocess_run_id", pa.string(), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("vector_id", pa.int64(), nullable=False),
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("frame_id", pa.int64(), nullable=False),
        pa.field("keyframe_seq", pa.int32(), nullable=False),
        pa.field("timestamp_ms", pa.int64(), nullable=False),
        pa.field("shot_id", pa.string(), nullable=False),
        pa.field("embedding_dim", pa.int32(), nullable=False),
        pa.field("vector_dtype", pa.string(), nullable=False),
        pa.field("l2_normalized", pa.bool_(), nullable=False),
        pa.field("keyframe_path", pa.string(), nullable=False),
        pa.field("created_at_utc", pa.string(), nullable=False),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_embedding_array(array: np.ndarray, expected_rows: int, expected_dim: int) -> None:
    if array.ndim != 2 or array.shape != (expected_rows, expected_dim):
        raise ContractError("embedding array shape does not match expected rows/dimension")
    if array.dtype != np.float32:
        raise ContractError("embedding array must use float32")
    if not np.isfinite(array).all():
        raise ContractError("embedding array contains NaN or infinity")
    norms = np.linalg.vector_norm(array, axis=1)
    if np.any(norms == 0) or not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
        raise ContractError("embedding vectors must be L2-normalized")


def write_embedding_shard(path: Path, array: np.ndarray) -> str:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    np.save(temp_path, array)
    saved_path = temp_path.with_suffix(temp_path.suffix + ".npy")
    os.replace(saved_path, path)
    return sha256_file(path)


def write_embedding_map_parquet(path: Path, records: Sequence[EmbeddingMapRecord]) -> str:
    columns = {field.name: [record.to_dict()[field.name] for record in records] for field in EMBEDDING_MAP_SCHEMA}
    table = pa.Table.from_pydict(columns, schema=EMBEDDING_MAP_SCHEMA)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temp_path)
    os.replace(temp_path, path)
    return sha256_file(path)


def write_json_atomically(path: Path, payload: dict[str, object]) -> str:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)
    return sha256_file(path)
