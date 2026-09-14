"""Production-facing WP03 artifact and four-model feedback adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from wp03.feedback_pool import EmbeddingReference, FeedbackPoolSnapshot, build_feedback_pool
from wp03.contracts import SearchCandidate, SearchRequest
from wp03.fusion import diversify_visual_candidates

from .contracts import CandidateId, CandidateMetadata, FeedbackValidationError, SessionPool, StableFeedbackConfig
from .ranking import fuse_embedding, late_rrf


class Wp03ArtifactVectorResolver:
    """Loads vectors by immutable WP03 embedding reference, with shard caching."""

    def __init__(self, artifact_root: Path, snapshot: FeedbackPoolSnapshot) -> None:
        self._root, self._snapshot, self._shards = artifact_root, snapshot, {}
        self._manifests = {model: json.loads((artifact_root / "manifests" / f"{model}.json").read_text(encoding="utf-8")) for model in snapshot.models}

    def __call__(self, ref: EmbeddingReference) -> np.ndarray:
        if ref.model_key not in self._manifests or ref.wp03_run_id != self._snapshot.wp03_run_id:
            raise FeedbackValidationError("embedding reference is outside the session snapshot")
        for shard in self._manifests[ref.model_key]["shards"]:
            if int(shard["record_start"]) <= ref.vector_id <= int(shard["record_end"]):
                key = (ref.model_key, int(shard["shard_id"]))
                vectors = self._shards.setdefault(key, np.load(self._root / "embeddings" / ref.model_key / f"shard-{key[1]:05d}.npy", mmap_mode="r"))
                return np.asarray(vectors[ref.vector_id - int(shard["record_start"])], dtype=np.float32)
        raise FeedbackValidationError("embedding vector is absent from WP03 shards")


class FourModelFeedbackRanker:
    """Applies composed fusion per WP03 model then late-RRFs the rankings."""

    def __init__(self, snapshot: FeedbackPoolSnapshot, encoders: Mapping[str, Callable[[tuple[str, ...]], np.ndarray]], resolver: Callable[[EmbeddingReference], np.ndarray], *, alpha: float = 0.75) -> None:
        if len(snapshot.models) != 4 or set(encoders) != set(snapshot.models):
            raise FeedbackValidationError("feedback ranker requires exactly the snapshotted four models")
        if not 0.0 <= alpha <= 1.0:
            raise FeedbackValidationError("feedback fusion alpha is invalid")
        self._snapshot, self._encoders, self._resolver, self._alpha = snapshot, dict(encoders), resolver, alpha
        self._by_id = {(item.candidate.video_id, item.candidate.frame_id): item for item in snapshot.candidates}

    def __call__(self, template: str, selected_video_id: str, selected_frame_id: int) -> tuple[CandidateId, ...]:
        selected = self._by_id.get((selected_video_id, selected_frame_id))
        if selected is None:
            raise FeedbackValidationError("selected frame is outside C0")
        rankings: dict[str, tuple[CandidateId, ...]] = {}
        for model in self._snapshot.models:
            text = np.asarray(self._encoders[model]((template,)), dtype=np.float32)
            if text.ndim != 2 or text.shape[0] != 1:
                raise FeedbackValidationError(f"{model} returned an invalid text embedding")
            fused = fuse_embedding(text[0], self._resolver(selected.embedding_refs[model]), alpha=self._alpha)
            scored = []
            for item in self._snapshot.candidates:
                score = float(np.dot(fused, self._resolver(item.embedding_refs[model])))
                scored.append((score, CandidateId(item.candidate.video_id, item.candidate.frame_id)))
            rankings[model] = tuple(candidate for _, candidate in sorted(scored, key=lambda value: (-value[0], value[1].video_id, value[1].frame_id)))
        return late_rrf(rankings, rrf_k=self._snapshot.rrf_k)

    def rank_for_session(self, _candidates: object, template: str, selected: CandidateId | None, _snapshot: object = None) -> tuple[CandidateId, ...]:
        """Adapter shape consumed by :class:`FeedbackSessions`."""
        if selected is None:
            raise FeedbackValidationError("feedback reranking requires a selected frame")
        return self(template, selected.video_id, selected.frame_id)


class Wp03FeedbackRuntime:
    """Thread-safe-enough wiring adapter for creating WP08 sessions from WP03."""

    def __init__(self, artifact_root: Path, encoders: Mapping[str, Callable[[tuple[str, ...]], np.ndarray]], *, stable_config: StableFeedbackConfig, pool_size: int = 500) -> None:
        if tuple(encoders) != stable_config.model_keys:
            raise FeedbackValidationError("runtime encoders do not match approved stable configuration")
        self._root, self._encoders, self._pool_size, self._config = artifact_root, dict(encoders), pool_size, stable_config
        self._snapshots: dict[tuple[CandidateId, ...], FeedbackPoolSnapshot] = {}

    def pool_provider(self, original_query: str) -> SessionPool:
        request = SearchRequest(query_id=f"wp08:{len(self._snapshots) + 1}", task="KIS", query_text=original_query, question=None, events=(), filters={}, limit=100, language=None, session_id=None)
        snapshot = build_feedback_pool(request=request, artifact_root=self._root, encoders=self._encoders, pool_size=self._pool_size, rrf_k=self._config.rrf_k)
        candidates = tuple(CandidateId(item.candidate.video_id, item.candidate.frame_id) for item in snapshot.candidates)
        self._snapshots[candidates] = snapshot
        return SessionPool(
            wp03_run_id=snapshot.wp03_run_id,
            candidates=candidates,
            candidate_metadata=tuple(
                CandidateMetadata(
                    candidate_id=CandidateId(item.candidate.video_id, item.candidate.frame_id),
                    timestamp_ms=item.candidate.timestamp_ms,
                    keyframe_path=_keyframe_path(item.candidate),
                )
                for item in snapshot.candidates
            ),
            snapshot=_serialize_snapshot(snapshot, self._config),
            provenance={
                "models": list(snapshot.models),
                "pool_size": snapshot.pool_size,
                "rrf_k": self._config.rrf_k,
                "fusion_alpha": self._config.fusion_alpha,
                "text_template_version": self._config.text_template_version,
                "diversity_policy": "wp03.diversify_visual_candidates:v1",
                "diversity_dedup_window_ms": self._config.diversity_dedup_window_ms,
                "benchmark_run_id": self._config.benchmark_run_id,
                "benchmark_approved_at_utc": self._config.approved_at_utc,
            },
        )

    def ranker(self, candidates: object, template: str, selected: CandidateId | None, serialized_snapshot: object) -> tuple[CandidateId, ...]:
        key = tuple(candidates)  # type: ignore[arg-type]
        snapshot = self._snapshots.get(key)
        if snapshot is None:
            snapshot = _deserialize_snapshot(serialized_snapshot)
            restored = tuple(CandidateId(item.candidate.video_id, item.candidate.frame_id) for item in snapshot.candidates)
            if restored != key:
                raise FeedbackValidationError("persisted WP03 snapshot does not match session C0")
            self._snapshots[key] = snapshot
        resolver = Wp03ArtifactVectorResolver(self._root, snapshot)
        return FourModelFeedbackRanker(snapshot, self._encoders, resolver, alpha=_feedback_alpha(serialized_snapshot)).rank_for_session(key, template, selected)

    def renderer(self, candidates: object, serialized_snapshot: object) -> tuple[CandidateId, ...]:
        """Apply the exact WP03 deterministic diversity implementation."""
        ranked = tuple(candidates)  # type: ignore[arg-type]
        candidate_set = set(ranked)
        snapshot = next((item for key, item in self._snapshots.items() if set(key) == candidate_set), None)
        if snapshot is None:
            snapshot = _deserialize_snapshot(serialized_snapshot)
            restored = tuple(CandidateId(item.candidate.video_id, item.candidate.frame_id) for item in snapshot.candidates)
            if set(restored) != candidate_set:
                raise FeedbackValidationError("persisted WP03 snapshot does not match rendered candidates")
            self._snapshots[restored] = snapshot
        by_id = {(item.candidate.video_id, item.candidate.frame_id): item.candidate for item in snapshot.candidates}
        ordered = tuple(by_id[(item.video_id, item.frame_id)] for item in ranked)
        dedup_window_ms = _diversity_dedup_window(serialized_snapshot)
        rendered = diversify_visual_candidates(ordered, limit=100, dedup_window_ms=dedup_window_ms)
        return tuple(CandidateId(item.video_id, item.frame_id) for item in rendered)


def _keyframe_path(candidate: SearchCandidate) -> str:
    for reference in candidate.evidence_refs:
        if reference.startswith("keyframe:"):
            return reference.removeprefix("keyframe:")
    raise FeedbackValidationError("WP03 candidate lacks a keyframe reference")


def _serialize_snapshot(snapshot: FeedbackPoolSnapshot, config: StableFeedbackConfig) -> dict[str, object]:
    """Serialize only immutable WP03 data needed for an exact post-restart rerank."""
    return {
        "query_id": snapshot.query_id,
        "wp03_run_id": snapshot.wp03_run_id,
        "models": list(snapshot.models),
        "pool_size": snapshot.pool_size,
        "rrf_k": snapshot.rrf_k,
        "feedback_fusion_alpha": config.fusion_alpha,
        "diversity_dedup_window_ms": config.diversity_dedup_window_ms,
        "benchmark_run_id": config.benchmark_run_id,
        "benchmark_approved_at_utc": config.approved_at_utc,
        "text_template_version": config.text_template_version,
        "candidates": [
            {
                "video_id": item.candidate.video_id,
                "frame_id": item.candidate.frame_id,
                "timestamp_ms": item.candidate.timestamp_ms,
                "keyframe_path": _keyframe_path(item.candidate),
                "fused_rank": item.fused_rank,
                "embedding_refs": {
                    model: {
                        "model_key": reference.model_key,
                        "vector_id": reference.vector_id,
                        "wp03_run_id": reference.wp03_run_id,
                        "mapping_sha256": reference.mapping_sha256,
                    }
                    for model, reference in item.embedding_refs.items()
                },
            }
            for item in snapshot.candidates
        ],
    }


def _deserialize_snapshot(value: object) -> FeedbackPoolSnapshot:
    try:
        raw = dict(value)  # type: ignore[arg-type]
        query_id, run_id = str(raw["query_id"]), str(raw["wp03_run_id"])
        models = tuple(str(model) for model in raw["models"])  # type: ignore[index]
        persisted_config = StableFeedbackConfig(
            model_keys=models,  # type: ignore[arg-type]
            benchmark_run_id=str(raw["benchmark_run_id"]),
            approved_at_utc=str(raw["benchmark_approved_at_utc"]),
            fusion_alpha=float(raw["feedback_fusion_alpha"]),
            rrf_k=int(raw["rrf_k"]),
            diversity_dedup_window_ms=int(raw["diversity_dedup_window_ms"]),
            text_template_version=str(raw["text_template_version"]),
        )
        candidates = []
        for item_value in raw["candidates"]:  # type: ignore[index]
            item = dict(item_value)
            refs = {
                str(model): EmbeddingReference(**dict(reference))
                for model, reference in dict(item["embedding_refs"]).items()
            }
            candidate = SearchCandidate.visual_rrf(
                query_id=query_id,
                event_index=None,
                preprocess_run_id="wp08-persisted-snapshot",
                video_id=str(item["video_id"]),
                frame_id=int(item["frame_id"]),
                timestamp_ms=int(item["timestamp_ms"]),
                rank=int(item["fused_rank"]),
                rrf_score=0.0,
                model_scores={},
                model_ranks={},
                keyframe_path=str(item["keyframe_path"]),
            )
            candidates.append(FeedbackPoolCandidate(candidate, int(item["fused_rank"]), refs))
        return FeedbackPoolSnapshot.create(
            query_id=query_id,
            wp03_run_id=run_id,
            models=models,
            pool_size=int(raw["pool_size"]),
            rrf_k=persisted_config.rrf_k,
            candidates=candidates,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FeedbackValidationError("persisted WP03 snapshot is invalid") from exc


def _feedback_alpha(value: object) -> float:
    try:
        alpha = float(dict(value).get("feedback_fusion_alpha", 0.75))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FeedbackValidationError("persisted feedback alpha is invalid") from exc
    if not 0.0 <= alpha <= 1.0:
        raise FeedbackValidationError("persisted feedback alpha is invalid")
    return alpha


def _diversity_dedup_window(value: object) -> int:
    try:
        window = int(dict(value).get("diversity_dedup_window_ms", 1_000))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FeedbackValidationError("persisted diversity configuration is invalid") from exc
    if window < 0:
        raise FeedbackValidationError("persisted diversity configuration is invalid")
    return window
