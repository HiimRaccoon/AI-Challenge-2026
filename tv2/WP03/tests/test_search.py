from __future__ import annotations

from pathlib import Path, PurePosixPath
import json

import numpy as np
import pytest

from wp03.corpus import load_corpus
from wp03.config import IndexConfig
from wp03.contracts import ContractError, SearchRequest
from wp03.mapping import load_search_mapping
from wp03.orchestrator import BuildRequest, build_model_artifacts
from wp03.search import (
    _expand_hits_with_shot_context, _merge_global_and_scoped_hits,
    search_image, search_visual, search_visual_batch,
)
from wp03.fusion import RankedHit
from tests.conftest import write_jsonl
from tests.test_corpus import record


class FakeEncoder:
    def encode_images(self, image_paths: tuple[Path, ...]) -> np.ndarray:
        return np.array([[1.0, 0.0] if path.name == "000003.jpg" else [0.0, 1.0] for path in image_paths], dtype=np.float32)

    def encode_text(self, texts: tuple[str, ...]) -> np.ndarray:
        return np.array([[0.0, 1.0] for _ in texts], dtype=np.float32)


def test_model_failure_reports_underlying_encoding_error(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1), FakeEncoder())
    class BrokenEncoder(FakeEncoder):
        def encode_text(self, texts):
            raise UnicodeEncodeError("utf-8", "\udc9d", 0, 1, "surrogates not allowed")
    with pytest.raises(ContractError, match="fake: UnicodeEncodeError"):
        search_visual(query_id="q-1", query_text="scene", event_index=None, artifact_root=artifact_root,
                      encoders={"fake": BrokenEncoder()}, requested_top_k=1,
                      candidate_k_per_model=1, hard_candidate_cap=None)


def test_search_returns_one_visual_response(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg"), record(42, "keyframes/L21_V001/000042.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = FakeEncoder()
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 2), encoder)

    response = search_visual(
        query_id="q-1",
        query_text="blue scene",
        event_index=None,
        artifact_root=artifact_root,
        encoders={"fake": encoder},
        requested_top_k=1,
        candidate_k_per_model=1,
        hard_candidate_cap=None,
    )

    assert response.degraded is False
    assert response.candidates[0].frame_id == 42
    assert response.candidates[0].source == "visual"


def test_search_consumes_pipeline_search_request(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = FakeEncoder()
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1), encoder)
    request = SearchRequest(
        query_id="q-1", task="KIS", query_text="blue scene", question=None,
        events=(), filters={}, limit=1, language="vi", session_id=None,
    )

    response = search_visual(
        request=request, artifact_root=artifact_root, encoders={"fake": encoder},
        candidate_k_per_model=1, hard_candidate_cap=None,
    )

    assert response.query_id == "q-1"
    assert response.requested_top_k == 1


def test_image_search_returns_the_same_visual_candidate_contract(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = FakeEncoder()
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1), encoder)
    request = SearchRequest(
        query_id="q-image", task="KIS", query_text="reference", question=None,
        events=(), filters={}, limit=1, language=None, session_id=None,
    )

    response = search_image(
        request=request, image_paths=(data_root / "keyframes/L21_V001/000003.jpg",), artifact_root=artifact_root,
        encoders={"fake": encoder}, candidate_k_per_model=1, hard_candidate_cap=None,
    )

    assert response.candidates[0].source == "visual"
    assert response.candidates[0].frame_id == 3


def test_search_excludes_model_with_incompatible_text_encoder(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    build_model_artifacts(
        BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1, "expected"), FakeEncoder()
    )

    class WrongEncoder(FakeEncoder):
        def compatibility_fingerprint(self) -> str:
            return "wrong"

    with pytest.raises(ContractError, match="no usable visual model"):
        search_visual(
            query_id="q-1", query_text="scene", event_index=None, artifact_root=artifact_root,
            encoders={"fake": WrongEncoder()}, requested_top_k=1, candidate_k_per_model=1, hard_candidate_cap=None,
        )


def test_cpu_fixture_search_uses_four_models_and_one_visual_list(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoders = {model: FakeEncoder() for model in ("beit3", "bge_vl", "metaclip2", "perception")}
    for model_key, encoder in encoders.items():
        build_model_artifacts(BuildRequest("run", model_key, f"{model_key}-rev", data_root, artifact_root, corpus, 1), encoder)

    response = search_visual(
        query_id="q-1", query_text="intersection", event_index=None, artifact_root=artifact_root,
        encoders=encoders, requested_top_k=3, candidate_k_per_model=20, hard_candidate_cap=None,
    )

    assert response.degraded is False
    assert response.models_used == ("beit3", "bge_vl", "metaclip2", "perception")
    assert {candidate.source for candidate in response.candidates} == {"visual"}


def test_search_rejects_manifest_with_invalid_mapping_digest(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = FakeEncoder()
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1), encoder)
    manifest_path = artifact_root / "manifests" / "fake.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mapping_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ContractError, match="mapping digest"):
        search_visual(
            query_id="q", query_text="scene", event_index=None, artifact_root=artifact_root,
            encoders={"fake": encoder}, requested_top_k=1, candidate_k_per_model=1, hard_candidate_cap=None,
        )


def test_search_uses_configured_rrf_constant(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = FakeEncoder()
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1), encoder)

    response = search_visual(
        query_id="q", query_text="scene", event_index=None, artifact_root=artifact_root,
        encoders={"fake": encoder}, requested_top_k=1, candidate_k_per_model=1,
        hard_candidate_cap=None, rrf_k=10, dedup_window_ms=None,
    )

    assert response.candidates[0].score == pytest.approx(1 / 11)


def test_batch_search_loads_each_faiss_index_once(data_root: Path, monkeypatch) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"
    encoder = FakeEncoder()
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 1), encoder)
    requests = tuple(
        SearchRequest(
            query_id=f"q-{index}", task="TRAKE", query_text=f"event {index}", question=None,
            events=(), filters={}, limit=1, language=None, session_id=None, event_index=index,
        )
        for index in range(3)
    )
    import wp03.search as search_module
    original_read_index = search_module.faiss.read_index
    read_count = 0

    def counted_read_index(path):
        nonlocal read_count
        read_count += 1
        return original_read_index(path)

    monkeypatch.setattr(search_module.faiss, "read_index", counted_read_index)

    responses = search_visual_batch(
        requests=requests, artifact_root=artifact_root, encoders={"fake": encoder},
        candidate_k_per_model=1, hard_candidate_cap=None,
    )
    repeated = search_visual_batch(
        requests=requests, artifact_root=artifact_root, encoders={"fake": encoder},
        candidate_k_per_model=1, hard_candidate_cap=None,
    )

    assert len(responses) == 3
    assert len(repeated) == 3
    assert read_count == 1


def test_mapping_stays_arrow_backed_and_shot_expansion_exact_rescores(data_root: Path) -> None:
    third = data_root / "keyframes" / "L21_V001" / "000004.jpg"
    third.write_bytes(b"image-four")
    first = record(3, "keyframes/L21_V001/000003.jpg")
    neighbor = record(4, "keyframes/L21_V001/000004.jpg")
    other = record(42, "keyframes/L21_V001/000042.jpg")
    other["shot_id"] = "L21_V001_S0002"
    write_jsonl(data_root / "frames.jsonl", [first, neighbor, other])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = data_root / "artifacts"

    class ShotEncoder(FakeEncoder):
        def encode_images(self, image_paths: tuple[Path, ...]) -> np.ndarray:
            values = {
                "000003.jpg": [1.0, 0.0],
                "000004.jpg": [0.8, 0.6],
                "000042.jpg": [0.6, 0.8],
            }
            return np.asarray([values[path.name] for path in image_paths], dtype=np.float32)

    build_model_artifacts(
        BuildRequest(
            "run", "fake", "rev", data_root, artifact_root, corpus, 3,
            index_config=IndexConfig(shot_expansion_per_hit=1),
        ),
        ShotEncoder(),
    )
    mapping = load_search_mapping(artifact_root / "embedding_maps" / "fake.parquet")
    index = __import__("faiss").read_index(str(artifact_root / "indexes" / "fake.faiss"))

    scores, ids = _expand_hits_with_shot_context(
        index=index,
        mapping=mapping,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        scores=np.asarray([1.0, 0.6], dtype=np.float32),
        ids=np.asarray([0, 2], dtype=np.int64),
        limit=2,
        neighbors_per_hit=1,
    )

    assert mapping.table.num_rows == 3
    assert mapping.has_shot_ids is True
    assert mapping.vector_ids_for_frames([{
        "video_id": "L21_V001", "frame_id": 4,
    }]) == (1,)
    assert ids.tolist() == [0, 1]
    assert scores.tolist() == pytest.approx([1.0, 0.8])


def test_scoped_frame_enters_deep_global_pool_without_hiding_global_anchor():
    global_hits = [
        RankedHit(index, f"V{index:03d}", index, index * 1000, f"{index}.jpg", index, 1 - index / 100)
        for index in range(1, 21)
    ]
    scoped = [RankedHit(99, "TARGET", 13718, 548720, "target.jpg", 1, 0.5)]

    merged = _merge_global_and_scoped_hits(
        global_hits, scoped, limit=20, rrf_k=60,
    )

    assert merged[0].video_id == "V001"
    assert any(hit.video_id == "TARGET" and hit.frame_id == 13718 for hit in merged)
