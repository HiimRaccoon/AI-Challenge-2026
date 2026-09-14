"""Sequential visual search over completed WP03 artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import faiss

from .artifacts import sha256_file
from .config import IndexConfig
from .contracts import ContractError, SearchCandidate, SearchRequest, SearchResponse
from .fusion import RankedHit, diversify_visual_candidates, fuse_rrf
from .index import configure_index_for_search, search_ip
from .mapping import SearchMapping, load_search_mapping


class SearchEncoder(Protocol):
    def encode_text(self, texts: tuple[str, ...]) -> np.ndarray: ...

    def compatibility_fingerprint(self) -> str: ...


class ImageSearchEncoder(SearchEncoder, Protocol):
    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray: ...


# Immutable, content-addressed search artifacts can stay in system RAM for a
# long-lived coordinator. FAISS indexes are CPU-side here, so this cache does
# not consume the 12 GB GPU budget. The manifest digests form part of the key;
# publishing a new artifact set naturally creates a new cache entry.
_ARTIFACT_CACHE: dict[
    tuple[str, str, str, str], tuple[faiss.Index, SearchMapping]
] = {}

_SCOPE_RRF_WEIGHT = 0.80


def _index_config(manifest: dict[str, object]) -> IndexConfig:
    return IndexConfig.from_mapping(manifest.get("index_config"))


def _expand_hits_with_shot_context(
    *, index: faiss.Index, mapping: SearchMapping, query_vector: np.ndarray,
    scores: np.ndarray, ids: np.ndarray, limit: int, neighbors_per_hit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-rescore temporal neighbors from shots reached by the ANN stage."""
    if neighbors_per_hit <= 0 or not mapping.has_shot_ids or not len(ids):
        return scores, ids
    candidate_ids = {int(vector_id) for vector_id in ids.tolist()}
    for vector_id in ids.tolist():
        candidate_ids.update(mapping.shot_neighbor_ids(int(vector_id), neighbors_per_hit))
    ordered_ids = np.asarray(sorted(candidate_ids), dtype=np.int64)
    vectors = np.vstack([index.reconstruct(int(vector_id)) for vector_id in ordered_ids]).astype(
        np.float32, copy=False,
    )
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.vector_norm(query))
    if not np.isfinite(query).all() or norm == 0:
        raise ContractError("query vector must be finite and non-zero")
    exact_scores = vectors @ (query / norm)
    order = np.lexsort((ordered_ids, -exact_scores))[:limit]
    return exact_scores[order], ordered_ids[order]


def _candidate_frame_scope(request: SearchRequest) -> tuple[dict[str, object], ...]:
    raw = request.filters.get("candidate_frames")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)) or len(raw) > 2_000:
        raise ContractError("candidate_frames must contain at most 2000 frame objects")
    frames: list[dict[str, object]] = []
    for value in raw:
        if not isinstance(value, dict):
            raise ContractError("candidate_frames must contain frame objects")
        video_id, frame_id = value.get("video_id"), value.get("frame_id")
        if (
            not isinstance(video_id, str) or not video_id
            or isinstance(frame_id, bool) or not isinstance(frame_id, int)
            or frame_id < 0
        ):
            raise ContractError("candidate_frames require video_id and non-negative frame_id")
        frames.append({"video_id": video_id, "frame_id": frame_id})
    return tuple(frames)


def _exact_scoped_hits(
    *, index: faiss.Index, mapping: SearchMapping, query_vector: np.ndarray,
    candidate_frames: Sequence[dict[str, object]], limit: int,
) -> list[RankedHit]:
    """Exactly rank a bounded, upstream-proven frame scope.

    Global HNSW retrieval remains the recall path.  This second score pass is
    only over canonical frames supplied by TV1/TV3, so a correct moment hidden
    below the corpus-wide ANN cutoff can still become the representative frame
    of a video already supported by transcript evidence.
    """
    vector_ids = mapping.vector_ids_for_frames(candidate_frames)
    if not vector_ids or limit <= 0:
        return []
    ids = np.asarray(vector_ids, dtype=np.int64)
    vectors = np.asarray(index.reconstruct_batch(ids), dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.vector_norm(query))
    if not np.isfinite(query).all() or norm == 0:
        raise ContractError("query vector must be finite and non-zero")
    scores = vectors @ (query / norm)
    order = np.lexsort((ids, -scores))[: min(limit, len(ids))]
    return [
        RankedHit(
            vector_id=int(ids[position]),
            video_id=str(mapping[int(ids[position])]["video_id"]),
            frame_id=int(mapping[int(ids[position])]["frame_id"]),
            timestamp_ms=int(mapping[int(ids[position])]["timestamp_ms"]),
            keyframe_path=str(mapping[int(ids[position])]["keyframe_path"]),
            rank=rank,
            similarity=float(scores[position]),
        )
        for rank, position in enumerate(order.tolist(), start=1)
    ]


def _merge_global_and_scoped_hits(
    global_hits: Sequence[RankedHit], scoped_hits: Sequence[RankedHit],
    *, limit: int, rrf_k: int,
) -> list[RankedHit]:
    """Blend a soft scoped rerank into global results without hiding either."""
    if not scoped_hits:
        return list(global_hits[:limit])
    merged: dict[tuple[str, int], dict[str, object]] = {}
    for weight, origin, hits in (
        (1.0, "global", global_hits),
        (_SCOPE_RRF_WEIGHT, "scope", scoped_hits),
    ):
        for hit in hits:
            key = (hit.video_id, hit.frame_id)
            entry = merged.setdefault(
                key, {"hit": hit, "score": 0.0, "origins": set()},
            )
            entry["score"] = float(entry["score"]) + weight / (rrf_k + hit.rank)
            origins = entry["origins"]
            assert isinstance(origins, set)
            origins.add(origin)
            if origin == "scope":
                entry["hit"] = hit
    ordered = sorted(
        merged.values(),
        key=lambda entry: (
            -float(entry["score"]),
            -len(entry["origins"]),
            entry["hit"].video_id,
            entry["hit"].frame_id,
        ),
    )[:limit]
    return [
        replace(entry["hit"], rank=rank)
        for rank, entry in enumerate(ordered, start=1)
    ]


def _multiscale_candidates(
    model_hits: dict[str, list[RankedHit]],
    *,
    request: SearchRequest,
    preprocess_run_id: str,
    candidate_k_per_model: int,
    rrf_k: int,
    dedup_window_ms: int | None,
) -> tuple[SearchCandidate, ...]:
    """Fuse both a precise shallow pool and a recall-oriented deep pool.

    Enlarging per-model FAISS depth can rescue rare frames, but candidates
    supported weakly by several models may then displace a strong one-model
    hit that appeared in the former shallow result. A second RRF over the two
    pool rankings preserves both signals without another encoder/GPU pass.
    """
    deep = fuse_rrf(
        model_hits,
        query_id=request.query_id,
        event_index=request.event_index,
        preprocess_run_id=preprocess_run_id,
        limit=max(request.limit, candidate_k_per_model),
        rrf_k=rrf_k,
    ).candidates
    shallow_depth = min(candidate_k_per_model, max(200, request.limit * 2))
    if shallow_depth >= candidate_k_per_model:
        candidates = deep
    else:
        shallow = fuse_rrf(
            {model: hits[:shallow_depth] for model, hits in model_hits.items()},
            query_id=request.query_id,
            event_index=request.event_index,
            preprocess_run_id=preprocess_run_id,
            limit=max(request.limit, shallow_depth),
            rrf_k=rrf_k,
        ).candidates
        merged: dict[tuple[str, int], dict[str, object]] = {}
        for pool_name, candidates_in_pool in (("shallow", shallow), ("deep", deep)):
            for candidate in candidates_in_pool:
                key = (candidate.video_id, candidate.frame_id)
                entry = merged.setdefault(
                    key,
                    {"candidate": candidate, "score": 0.0, "pools": set()},
                )
                entry["score"] = float(entry["score"]) + 1.0 / (rrf_k + candidate.rank)
                pools = entry["pools"]
                assert isinstance(pools, set)
                pools.add(pool_name)
                if pool_name == "deep":
                    entry["candidate"] = candidate
        ordered = sorted(
            merged.values(),
            key=lambda entry: (
                -float(entry["score"]),
                -len(entry["pools"]),
                entry["candidate"].video_id,
                entry["candidate"].frame_id,
            ),
        )
        candidates = tuple(
            replace(
                entry["candidate"],
                rank=rank,
                score=float(entry["score"]),
                matched_filters=tuple(
                    (*entry["candidate"].matched_filters, *(
                        f"candidate_pool:{pool}" for pool in sorted(entry["pools"])
                    ))
                ),
            )
            for rank, entry in enumerate(ordered, start=1)
        )
    return (
        tuple(candidates[:request.limit])
        if dedup_window_ms is None
        else diversify_visual_candidates(
            candidates, limit=request.limit, dedup_window_ms=dedup_window_ms,
        )
    )


def _load_search_artifacts(
    *, model_key: str, index_path: Path, mapping_path: Path,
    manifest: dict[str, object],
) -> tuple[faiss.Index, SearchMapping]:
    index_digest = str(manifest.get("index_sha256", ""))
    mapping_digest = str(manifest.get("mapping_sha256", ""))
    index_config = _index_config(manifest)
    config_key = json.dumps(index_config.to_dict(), sort_keys=True)
    key = (str(index_path.resolve()), index_digest, mapping_digest, config_key)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached
    if sha256_file(index_path) != index_digest:
        raise ContractError(f"{model_key} index digest mismatch")
    if sha256_file(mapping_path) != mapping_digest:
        raise ContractError(f"{model_key} mapping digest mismatch")
    mapping = load_search_mapping(mapping_path)
    index = faiss.read_index(str(index_path))
    if index.ntotal != len(mapping) or index.ntotal != manifest.get("vector_count"):
        raise ContractError(f"{model_key} index/map vector count mismatch")
    configure_index_for_search(index, index_config)
    _ARTIFACT_CACHE[key] = (index, mapping)
    return index, mapping


def search_visual(
    *,
    request: SearchRequest | None = None,
    artifact_root: Path,
    encoders: dict[str, SearchEncoder],
    candidate_k_per_model: int,
    hard_candidate_cap: int | None,
    query_id: str | None = None,
    query_text: str | None = None,
    event_index: int | None = None,
    requested_top_k: int | None = None,
    internal_pool: bool = False,
    rrf_k: int = 60,
    dedup_window_ms: int | None = 1_000,
) -> SearchResponse:
    """Search usable model indexes one at a time and RRF their ranked hits."""

    if request is None:
        if query_id is None or query_text is None or requested_top_k is None:
            raise ContractError("request or legacy query fields are required")
        request = SearchRequest(
            query_id=query_id,
            task="KIS",
            query_text=query_text,
            question=None,
            events=(),
            filters={},
            limit=requested_top_k,
            language=None,
            session_id=None,
            event_index=event_index,
            internal_pool=internal_pool,
        )
    normalized_query = request.visual_query_text()
    return _search_embeddings(
        request=request,
        artifact_root=artifact_root,
        encoders=encoders,
        candidate_k_per_model=candidate_k_per_model,
        hard_candidate_cap=hard_candidate_cap,
        rrf_k=rrf_k,
        dedup_window_ms=dedup_window_ms,
        encode_query=lambda encoder: encoder.encode_text((normalized_query,)),
    )


def search_visual_batch(
    *,
    requests: Sequence[SearchRequest],
    artifact_root: Path,
    encoders: dict[str, SearchEncoder],
    candidate_k_per_model: int,
    hard_candidate_cap: int | None,
    rrf_k: int = 60,
    dedup_window_ms: int | None = 1_000,
) -> tuple[SearchResponse, ...]:
    """Search several text queries while loading each visual index once.

    TRAKE queries a whole sequence plus every ordered event.  Executing them
    as independent processes used to reload the same FAISS indexes and model
    workers repeatedly.  This keeps every response's public candidate
    contract while encoding all query texts in one worker request per model.
    """
    if not requests:
        return ()
    if candidate_k_per_model <= 0 or rrf_k <= 0:
        raise ContractError("candidate_k_per_model and rrf_k must be positive")
    if dedup_window_ms is not None and dedup_window_ms < 0:
        raise ContractError("dedup_window_ms must be non-negative or null")

    normalized_queries = tuple(request.visual_query_text() for request in requests)
    requested_models = tuple(encoders)
    model_hits_by_request: list[dict[str, list[RankedHit]]] = [dict() for _ in requests]
    used_models: list[str] = []
    failures: list[str] = []
    run_id: str | None = None
    preprocess_run_id: str | None = None

    for model_key, encoder in encoders.items():
        manifest_path = artifact_root / "manifests" / f"{model_key}.json"
        index_path = artifact_root / "indexes" / f"{model_key}.faiss"
        mapping_path = artifact_root / "embedding_maps" / f"{model_key}.parquet"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        index, mapping = _load_search_artifacts(
            model_key=model_key, index_path=index_path, mapping_path=mapping_path,
            manifest=manifest,
        )
        try:
            required_compatibility = str(manifest.get("compatibility_fingerprint", ""))
            fingerprint = getattr(encoder, "compatibility_fingerprint", None)
            if required_compatibility and (not callable(fingerprint) or fingerprint() != required_compatibility):
                continue
            query_vectors = np.asarray(encoder.encode_text(normalized_queries), dtype=np.float32)
            if query_vectors.ndim != 2 or query_vectors.shape[0] != len(requests):
                raise ContractError("query encoder must return one vector per query")
            for request_index, request in enumerate(requests):
                limit = min(len(mapping), max(request.limit, candidate_k_per_model))
                if hard_candidate_cap is not None:
                    limit = min(limit, hard_candidate_cap)
                scores, ids = search_ip(index, query_vectors[request_index], limit)
                scores, ids = _expand_hits_with_shot_context(
                    index=index, mapping=mapping,
                    query_vector=query_vectors[request_index], scores=scores, ids=ids,
                    limit=limit,
                    neighbors_per_hit=_index_config(manifest).shot_expansion_per_hit,
                )
                global_hits = [
                    RankedHit(
                        vector_id=vector_id,
                        video_id=str(mapping[vector_id]["video_id"]),
                        frame_id=int(mapping[vector_id]["frame_id"]),
                        timestamp_ms=int(mapping[vector_id]["timestamp_ms"]),
                        keyframe_path=str(mapping[vector_id]["keyframe_path"]),
                        rank=rank,
                        similarity=float(score),
                    )
                    for rank, (score, vector_id) in enumerate(zip(scores.tolist(), ids.tolist()), start=1)
                ]
                scope = _candidate_frame_scope(request)
                scoped_hits = _exact_scoped_hits(
                    index=index, mapping=mapping,
                    query_vector=query_vectors[request_index],
                    candidate_frames=scope, limit=limit,
                ) if scope else []
                model_hits_by_request[request_index][model_key] = _merge_global_and_scoped_hits(
                    global_hits, scoped_hits, limit=limit, rrf_k=rrf_k,
                )
        except (OSError, ValueError, ContractError) as exc:
            failures.append(f"{model_key}: {type(exc).__name__}: {str(exc)[:500]}")
            continue
        run_id = str(manifest["wp03_run_id"])
        preprocess_run_id = str(manifest["preprocess_run_id"])
        used_models.append(model_key)

    if not used_models or run_id is None or preprocess_run_id is None:
        raise ContractError("no usable visual model" + ("; " + "; ".join(failures) if failures else "; check model manifest compatibility"))

    responses: list[SearchResponse] = []
    for request, model_hits in zip(requests, model_hits_by_request):
        candidates = _multiscale_candidates(
            model_hits,
            request=request,
            preprocess_run_id=preprocess_run_id,
            candidate_k_per_model=candidate_k_per_model,
            rrf_k=rrf_k,
            dedup_window_ms=dedup_window_ms,
        )
        responses.append(
            SearchResponse.create(
                query_id=request.query_id,
                wp03_run_id=run_id,
                preprocess_run_id=preprocess_run_id,
                requested_top_k=request.limit,
                candidate_k_per_model=candidate_k_per_model,
                hard_candidate_cap=hard_candidate_cap,
                models_requested=requested_models,
                models_used=tuple(used_models),
                candidates=candidates,
            )
        )
    return tuple(responses)


def search_image(
    *,
    request: SearchRequest,
    image_paths: Sequence[Path],
    artifact_root: Path,
    encoders: dict[str, ImageSearchEncoder],
    candidate_k_per_model: int,
    hard_candidate_cap: int | None,
    rrf_k: int = 60,
    dedup_window_ms: int | None = 1_000,
) -> SearchResponse:
    """Retrieve frames from one or more reference images in each model's own space."""

    if not image_paths:
        raise ContractError("image_paths must not be empty")
    return _search_embeddings(
        request=request,
        artifact_root=artifact_root,
        encoders=encoders,
        candidate_k_per_model=candidate_k_per_model,
        hard_candidate_cap=hard_candidate_cap,
        rrf_k=rrf_k,
        dedup_window_ms=dedup_window_ms,
        encode_query=lambda encoder: encoder.encode_images(tuple(image_paths)),
    )


def _search_embeddings(
    *,
    request: SearchRequest,
    artifact_root: Path,
    encoders: dict[str, SearchEncoder],
    candidate_k_per_model: int,
    hard_candidate_cap: int | None,
    rrf_k: int,
    dedup_window_ms: int | None,
    encode_query: Callable[[SearchEncoder], np.ndarray],
) -> SearchResponse:
    requested_top_k = request.limit
    if requested_top_k <= 0 or candidate_k_per_model <= 0 or rrf_k <= 0:
        raise ContractError("requested_top_k, candidate_k_per_model and rrf_k must be positive")
    if dedup_window_ms is not None and dedup_window_ms < 0:
        raise ContractError("dedup_window_ms must be non-negative or null")
    requested_models = tuple(encoders)
    model_hits: dict[str, list[RankedHit]] = {}
    used_models: list[str] = []
    failures: list[str] = []
    run_id: str | None = None
    preprocess_run_id: str | None = None
    for model_key, encoder in encoders.items():
        manifest_path = artifact_root / "manifests" / f"{model_key}.json"
        index_path = artifact_root / "indexes" / f"{model_key}.faiss"
        mapping_path = artifact_root / "embedding_maps" / f"{model_key}.parquet"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        index, mapping = _load_search_artifacts(
            model_key=model_key, index_path=index_path, mapping_path=mapping_path,
            manifest=manifest,
        )
        try:
            required_compatibility = str(manifest.get("compatibility_fingerprint", ""))
            fingerprint = getattr(encoder, "compatibility_fingerprint", None)
            if required_compatibility and (not callable(fingerprint) or fingerprint() != required_compatibility):
                continue
            query_vectors = np.asarray(encode_query(encoder), dtype=np.float32)
            if query_vectors.ndim != 2 or query_vectors.shape[0] == 0:
                raise ContractError("query encoder must return a non-empty matrix")
            query_vector = query_vectors.mean(axis=0)
            limit = min(len(mapping), max(requested_top_k, candidate_k_per_model))
            if hard_candidate_cap is not None:
                limit = min(limit, hard_candidate_cap)
            scores, ids = search_ip(index, query_vector, limit)
            scores, ids = _expand_hits_with_shot_context(
                index=index, mapping=mapping, query_vector=query_vector,
                scores=scores, ids=ids, limit=limit,
                neighbors_per_hit=_index_config(manifest).shot_expansion_per_hit,
            )
        except (OSError, ValueError, ContractError) as exc:
            failures.append(f"{model_key}: {type(exc).__name__}: {str(exc)[:500]}")
            continue
        run_id = str(manifest["wp03_run_id"])
        preprocess_run_id = str(manifest["preprocess_run_id"])
        hits: list[RankedHit] = []
        for rank, (score, vector_id) in enumerate(zip(scores.tolist(), ids.tolist()), start=1):
            row = mapping[vector_id]
            hits.append(
                RankedHit(
                    vector_id=vector_id,
                    video_id=str(row["video_id"]),
                    frame_id=int(row["frame_id"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    keyframe_path=str(row["keyframe_path"]),
                    rank=rank,
                    similarity=float(score),
                )
            )
        scope = _candidate_frame_scope(request)
        scoped_hits = _exact_scoped_hits(
            index=index, mapping=mapping, query_vector=query_vector,
            candidate_frames=scope, limit=limit,
        ) if scope else []
        hits = _merge_global_and_scoped_hits(
            hits, scoped_hits, limit=limit, rrf_k=rrf_k,
        )
        model_hits[model_key] = hits
        used_models.append(model_key)
    if not used_models or run_id is None or preprocess_run_id is None:
        raise ContractError("no usable visual model" + ("; " + "; ".join(failures) if failures else "; check model manifest compatibility"))
    candidates = _multiscale_candidates(
        model_hits,
        request=request,
        preprocess_run_id=preprocess_run_id,
        candidate_k_per_model=candidate_k_per_model,
        rrf_k=rrf_k,
        dedup_window_ms=dedup_window_ms,
    )
    return SearchResponse.create(
        query_id=request.query_id,
        wp03_run_id=run_id,
        preprocess_run_id=preprocess_run_id,
        requested_top_k=requested_top_k,
        candidate_k_per_model=candidate_k_per_model,
        hard_candidate_cap=hard_candidate_cap,
        models_requested=requested_models,
        models_used=tuple(used_models),
        candidates=candidates,
    )
