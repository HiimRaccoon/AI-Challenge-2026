"""Exact, streaming FAISS index construction and querying."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from .artifacts import sha256_file, validate_embedding_array
from .config import IndexConfig
from .contracts import ContractError


@dataclass(frozen=True)
class IndexBuildReport:
    vector_count: int
    dimension: int
    index_sha256: str


def build_ip_index(
    shard_paths: Sequence[Path], dimension: int, output_path: Path,
    *, config: IndexConfig | None = None, block_rows: int = 4096,
) -> IndexBuildReport:
    """Build one cosine index without concatenating embedding shards."""

    if dimension <= 0 or block_rows <= 0:
        raise ContractError("dimension and block_rows must be positive")
    spec = config or IndexConfig()
    if spec.type == "flat_ip":
        index: faiss.Index = faiss.IndexFlatIP(dimension)
    elif spec.type == "hnsw_flat":
        index = faiss.IndexHNSWFlat(dimension, spec.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = spec.ef_construction
        index.hnsw.efSearch = spec.ef_search
    else:  # IndexConfig validates this; keep the boundary defensive.
        raise ContractError(f"unsupported index type: {spec.type}")
    for shard_path in shard_paths:
        shard = np.load(shard_path, mmap_mode="r")
        if shard.ndim != 2 or shard.shape[1] != dimension:
            raise ContractError("embedding shard dimension does not match index")
        for start in range(0, shard.shape[0], block_rows):
            block = np.asarray(shard[start : start + block_rows], dtype=np.float32)
            validate_embedding_array(block, expected_rows=block.shape[0], expected_dim=dimension)
            index.add(block)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    faiss.write_index(index, str(temp_path))
    os.replace(temp_path, output_path)
    return IndexBuildReport(vector_count=index.ntotal, dimension=dimension, index_sha256=sha256_file(output_path))


def build_flat_ip_index(
    shard_paths: Sequence[Path], dimension: int, output_path: Path, block_rows: int = 4096
) -> IndexBuildReport:
    """Backward-compatible exact-index builder."""
    return build_ip_index(
        shard_paths, dimension, output_path,
        config=IndexConfig(type="flat_ip"), block_rows=block_rows,
    )


def configure_index_for_search(index: faiss.Index, config: IndexConfig) -> None:
    """Apply mutable query-time settings after deserializing an index."""
    if config.type == "hnsw_flat":
        hnsw = getattr(index, "hnsw", None)
        if hnsw is None:
            raise ContractError("manifest declares hnsw_flat but artifact is not HNSW")
        hnsw.efSearch = config.ef_search


def search_ip(index: faiss.Index, query_vector: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Search a loaded IP index and deterministically order returned scores."""
    if limit <= 0:
        raise ContractError("limit must be positive")
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != index.d:
        raise ContractError("query vector dimension does not match index")
    norm = float(np.linalg.vector_norm(query))
    if not np.isfinite(query).all() or norm == 0:
        raise ContractError("query vector must be finite and non-zero")
    query /= norm
    count = min(limit, index.ntotal)
    scores, ids = index.search(query, count)
    score_row, id_row = scores[0], ids[0]
    valid = id_row >= 0
    score_row, id_row = score_row[valid], id_row[valid]
    order = np.lexsort((id_row, -score_row))
    return score_row[order], id_row[order]


def search_flat_ip(index: faiss.Index, query_vector: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for callers using the old function name."""
    return search_ip(index, query_vector, limit)


def search_flat_ip_index(index_path: Path, query_vector: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for callers that do not already own an index."""
    return search_ip(faiss.read_index(str(index_path)), query_vector, limit)
