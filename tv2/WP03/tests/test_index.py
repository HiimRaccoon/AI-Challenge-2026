from __future__ import annotations

from pathlib import Path

import numpy as np
import faiss
import pytest

from wp03.config import IndexConfig
from wp03.index import build_flat_ip_index, build_ip_index, search_flat_ip, search_flat_ip_index


def test_streaming_index_returns_the_vector_from_second_shard(tmp_path: Path) -> None:
    first = tmp_path / "a.npy"
    second = tmp_path / "b.npy"
    np.save(first, np.array([[1.0, 0.0]], dtype=np.float32))
    np.save(second, np.array([[0.0, 1.0]], dtype=np.float32))

    report = build_flat_ip_index((first, second), 2, tmp_path / "index.faiss")
    scores, ids = search_flat_ip_index(tmp_path / "index.faiss", np.array([0.0, 1.0], np.float32), 1)

    assert report.vector_count == 2
    assert scores.tolist() == [1.0]
    assert ids.tolist() == [1]


def test_loaded_index_can_serve_repeated_queries_without_reload(tmp_path: Path) -> None:
    shard = tmp_path / "vectors.npy"
    index_path = tmp_path / "index.faiss"
    np.save(shard, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    build_flat_ip_index((shard,), 2, index_path)
    index = faiss.read_index(str(index_path))

    _, first_ids = search_flat_ip(index, np.array([1.0, 0.0], np.float32), 1)
    _, second_ids = search_flat_ip(index, np.array([0.0, 1.0], np.float32), 1)

    assert first_ids.tolist() == [0]
    assert second_ids.tolist() == [1]


def test_hnsw_index_returns_nearest_normalized_vector(tmp_path: Path) -> None:
    shard = tmp_path / "vectors.npy"
    index_path = tmp_path / "hnsw.faiss"
    vectors = np.eye(8, dtype=np.float32)
    np.save(shard, vectors)

    report = build_ip_index(
        (shard,), 8, index_path,
        config=IndexConfig(
            type="hnsw_flat", hnsw_m=4, ef_construction=40, ef_search=32,
        ),
    )
    index = faiss.read_index(str(index_path))
    scores, ids = search_flat_ip(index, vectors[5], 1)

    assert report.vector_count == 8
    assert ids.tolist() == [5]
    assert scores.tolist() == pytest.approx([1.0])
