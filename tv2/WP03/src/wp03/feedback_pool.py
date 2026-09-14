"""Internal, immutable candidate pools used by WP08 feedback sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import faiss
import numpy as np

from .artifacts import sha256_file
from .config import IndexConfig
from .contracts import ContractError, SearchCandidate, SearchRequest
from .fusion import RankedHit, fuse_rrf
from .index import configure_index_for_search, search_ip
from .mapping import SearchMapping, load_search_mapping


@dataclass(frozen=True)
class EmbeddingReference:
    """Stable address of one frame embedding in one immutable WP03 run."""

    model_key: str
    vector_id: int
    wp03_run_id: str
    mapping_sha256: str

    def __post_init__(self) -> None:
        if not self.model_key or self.vector_id < 0 or not self.wp03_run_id:
            raise ContractError("embedding reference is invalid")
        if len(self.mapping_sha256) != 64:
            raise ContractError("embedding reference mapping digest is invalid")


@dataclass(frozen=True)
class FeedbackPoolCandidate:
    candidate: SearchCandidate
    fused_rank: int
    embedding_refs: Mapping[str, EmbeddingReference]

    def __post_init__(self) -> None:
        if self.fused_rank < 1:
            raise ContractError("fused rank must be positive")
        refs = dict(self.embedding_refs)
        if any(key != ref.model_key for key, ref in refs.items()):
            raise ContractError("embedding reference model keys must match")
        object.__setattr__(self, "embedding_refs", refs)


@dataclass(frozen=True)
class FeedbackPoolSnapshot:
    query_id: str
    wp03_run_id: str
    models: tuple[str, ...]
    pool_size: int
    rrf_k: int
    candidates: tuple[FeedbackPoolCandidate, ...]

    @classmethod
    def create(
        cls,
        *,
        query_id: str,
        wp03_run_id: str,
        models: Sequence[str],
        pool_size: int,
        rrf_k: int,
        candidates: Sequence[FeedbackPoolCandidate],
    ) -> "FeedbackPoolSnapshot":
        model_tuple = tuple(models)
        candidate_tuple = tuple(candidates)
        if not query_id or not wp03_run_id or not model_tuple or len(set(model_tuple)) != len(model_tuple):
            raise ContractError("feedback pool metadata is invalid")
        if pool_size < 1 or rrf_k < 1 or len(candidate_tuple) > pool_size:
            raise ContractError("feedback pool limits are invalid")
        identities = {(item.candidate.video_id, item.candidate.frame_id) for item in candidate_tuple}
        if len(identities) != len(candidate_tuple):
            raise ContractError("feedback pool contains duplicate candidates")
        if tuple(item.fused_rank for item in candidate_tuple) != tuple(range(1, len(candidate_tuple) + 1)):
            raise ContractError("feedback pool fused ranks must be contiguous")
        for item in candidate_tuple:
            if set(item.embedding_refs) != set(model_tuple):
                raise ContractError("feedback pool candidate lacks embedding references")
            if any(ref.wp03_run_id != wp03_run_id for ref in item.embedding_refs.values()):
                raise ContractError("feedback pool embedding run does not match snapshot")
        return cls(query_id, wp03_run_id, model_tuple, pool_size, rrf_k, candidate_tuple)


def build_feedback_pool(
    *, request: SearchRequest, artifact_root: Path, encoders: Mapping[str, object], pool_size: int = 500, rrf_k: int = 60
) -> FeedbackPoolSnapshot:
    """Build WP08's private raw pool without using the public result limit."""
    if pool_size < 1 or rrf_k < 1 or len(encoders) != 4:
        raise ContractError("feedback pool requires four models and positive limits")
    query = request.visual_query_text()
    maps: dict[str, SearchMapping] = {}
    manifests: dict[str, dict[str, object]] = {}
    model_hits: dict[str, list[RankedHit]] = {}
    for model_key, encoder in encoders.items():
        manifest_path = artifact_root / "manifests" / f"{model_key}.json"
        mapping_path = artifact_root / "embedding_maps" / f"{model_key}.parquet"
        index_path = artifact_root / "indexes" / f"{model_key}.faiss"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or sha256_file(mapping_path) != manifest.get("mapping_sha256") or sha256_file(index_path) != manifest.get("index_sha256"):
            raise ContractError(f"{model_key} artifact is invalid")
        mapping = load_search_mapping(mapping_path)
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(mapping):
            raise ContractError(f"{model_key} index/map vector count mismatch")
        configure_index_for_search(
            index, IndexConfig.from_mapping(manifest.get("index_config")),
        )
        encode = getattr(encoder, "encode_text", None)
        if not callable(encode):
            raise ContractError(f"{model_key} lacks text encoder")
        vectors = np.asarray(encode((query,)), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != 1:
            raise ContractError(f"{model_key} returned invalid query embedding")
        scores, ids = search_ip(index, vectors[0], min(pool_size, len(mapping)))
        model_hits[model_key] = [RankedHit(int(vector_id), str(mapping[int(vector_id)]["video_id"]), int(mapping[int(vector_id)]["frame_id"]), int(mapping[int(vector_id)]["timestamp_ms"]), str(mapping[int(vector_id)]["keyframe_path"]), rank, float(score)) for rank, (score, vector_id) in enumerate(zip(scores.tolist(), ids.tolist()), 1)]
        maps[model_key], manifests[model_key] = mapping, manifest
    run_ids = {str(item["wp03_run_id"]) for item in manifests.values()}
    prep_ids = {str(item["preprocess_run_id"]) for item in manifests.values()}
    if len(run_ids) != 1 or len(prep_ids) != 1:
        raise ContractError("feedback pool model runs are inconsistent")
    fused = fuse_rrf(model_hits, query_id=request.query_id, event_index=request.event_index, preprocess_run_id=prep_ids.pop(), limit=pool_size, rrf_k=rrf_k).candidates
    identities = {(candidate.video_id, candidate.frame_id) for candidate in fused}
    lookups: dict[str, dict[tuple[str, int], int]] = {}
    for model, mapping in maps.items():
        lookup: dict[tuple[str, int], int] = {}
        for vector_id in range(len(mapping)):
            row = mapping[vector_id]
            identity = (str(row["video_id"]), int(row["frame_id"]))
            if identity in identities:
                lookup[identity] = vector_id
        lookups[model] = lookup
    run_id = run_ids.pop()
    candidates = []
    for fused_rank, candidate in enumerate(fused, 1):
        identity = (candidate.video_id, candidate.frame_id)
        try:
            refs = {model: EmbeddingReference(model, lookup[identity], run_id, str(manifests[model]["mapping_sha256"])) for model, lookup in lookups.items()}
        except KeyError as exc:
            raise ContractError("feedback pool candidate lacks embedding references") from exc
        candidates.append(FeedbackPoolCandidate(candidate, fused_rank, refs))
    return FeedbackPoolSnapshot.create(query_id=request.query_id, wp03_run_id=run_id, models=tuple(encoders), pool_size=pool_size, rrf_k=rrf_k, candidates=candidates)
