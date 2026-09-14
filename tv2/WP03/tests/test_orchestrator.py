from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from wp03.config import IndexConfig
from wp03.corpus import load_corpus
from wp03.orchestrator import BuildRequest, build_all_models, build_model_artifacts
from tests.conftest import write_jsonl
from tests.test_corpus import record


class FakeImageEncoder:
    def encode_images(self, image_paths: tuple[Path, ...]) -> np.ndarray:
        rows = np.array([[1.0, 0.0] if path.name == "000003.jpg" else [0.0, 1.0] for path in image_paths], dtype=np.float32)
        return rows


class CountingEncoder(FakeImageEncoder):
    def __init__(self) -> None:
        self.calls = 0

    def encode_images(self, image_paths: tuple[Path, ...]) -> np.ndarray:
        self.calls += 1
        return super().encode_images(image_paths)


def test_build_model_writes_shards_mapping_index_and_complete_manifest(data_root: Path) -> None:
    write_jsonl(
        data_root / "frames.jsonl",
        [record(3, "keyframes/L21_V001/000003.jpg"), record(42, "keyframes/L21_V001/000042.jpg")],
    )
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"

    manifest = build_model_artifacts(
        BuildRequest(
            run_id="wp03-run",
            model_key="fake",
            model_version="test-revision",
            data_root=data_root,
            artifact_root=artifact_root,
            corpus=corpus,
            shard_size=1,
        ),
        FakeImageEncoder(),
    )

    assert manifest["status"] == "complete"
    assert manifest["vector_count"] == 2
    assert (artifact_root / "embedding_maps" / "fake.parquet").exists()
    assert (artifact_root / "indexes" / "fake.faiss").exists()


def test_resume_reuses_valid_shards_and_rebuilds_only_a_corrupt_shard(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg"), record(42, "keyframes/L21_V001/000042.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = CountingEncoder()
    build = BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1)
    build_model_artifacts(build, encoder)
    assert encoder.calls == 2

    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1, resume=True), encoder)
    assert encoder.calls == 2

    (artifact_root / "embeddings" / "fake" / "shard-00001.npy").write_bytes(b"corrupt")
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1, resume=True), encoder)
    assert encoder.calls == 3


def test_resume_can_rebuild_index_policy_without_reencoding_shards(data_root: Path) -> None:
    write_jsonl(
        data_root / "frames.jsonl",
        [record(3, "keyframes/L21_V001/000003.jpg"), record(42, "keyframes/L21_V001/000042.jpg")],
    )
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = CountingEncoder()
    build_model_artifacts(
        BuildRequest(
            "run", "fake", "rev", data_root, artifact_root, corpus, 1,
            config_digest="flat-config",
        ),
        encoder,
    )

    manifest = build_model_artifacts(
        BuildRequest(
            "run", "fake", "rev", data_root, artifact_root, corpus, 1,
            resume=True, config_digest="hnsw-config",
            index_config=IndexConfig(
                type="hnsw_flat", hnsw_m=4, ef_construction=40, ef_search=32,
            ),
        ),
        encoder,
    )

    assert encoder.calls == 2
    assert manifest["index_config"]["type"] == "hnsw_flat"


def test_completed_run_rejects_overwrite_without_resume(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    build = BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1)
    build_model_artifacts(build, FakeImageEncoder())

    with pytest.raises(ValueError, match="resume"):
        build_model_artifacts(build, FakeImageEncoder())


def test_build_all_models_returns_degraded_summary_when_one_model_fails(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"

    class FailingEncoder:
        def encode_images(self, image_paths):
            raise RuntimeError("backend failed")

    summary = build_all_models(
        (
            BuildRequest("run", "good", "rev", data_root, artifact_root, corpus, 1),
            BuildRequest("run", "bad", "rev", data_root, artifact_root, corpus, 1),
        ),
        {"good": FakeImageEncoder(), "bad": FailingEncoder()},
    )

    assert summary.status == "complete"
    assert summary.degraded is True
    assert summary.models["good"]["status"] == "complete"
    assert summary.models["bad"]["status"] == "failed"
    assert (artifact_root / "reports" / "build-summary.json").is_file()
