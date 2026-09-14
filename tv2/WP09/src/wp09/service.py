"""Refinement orchestration with explicit fatal/degraded semantics."""

from __future__ import annotations

from dataclasses import replace
from time import monotonic
from typing import Sequence

from .cache import CachedWindow, DecodedWindowCache, DecodedWindowKey, WindowRequestKey
from .config import RefinementConfig
from .contracts import (
    ContractError, ExactFrameHypothesis, RefineRequest, RefineResult, RefinementAudit,
    RefinementStatus, RefinementUnavailable,
)
from .decoder import DecodeBudgetExhausted, DecodedFrame, LocalWindow, SamplingResult, VideoDecoder, sample_two_stage
from .policies import ScoredFrame, select_hypotheses
from .scoring import FrameScorer, ScoringUnavailable


class ExactFrameRefiner:
    """A local-only original-video refiner; it never retrieves or submits."""

    def __init__(self, decoder: VideoDecoder, scorer: FrameScorer | None, config: RefinementConfig, *, cache: DecodedWindowCache | None = None) -> None:
        if not getattr(decoder, "mapping_guaranteed", False):
            raise ContractError("decoder must provide a canonical mapping guarantee")
        self._decoder = decoder
        self._scorer = scorer
        self._config = config
        self._cache = cache or DecodedWindowCache(config.cache_max_entries, config.cache_ttl_seconds)

    def radius_for(self, request: RefineRequest) -> int:
        return self._config.radius_for_confidence(request.candidate.confidence, request.decode_budget)

    def refine(self, request: RefineRequest) -> RefineResult:
        started = monotonic()
        radius = self.radius_for(request)
        request_key = self._request_cache_key(request, radius)
        cached = self._cache.get_for_request(request_key)
        if cached is not None:
            best = next((frame for frame in cached.frames if frame.frame_id == cached.best_frame_id), cached.frames[0])
            sampled = SamplingResult(LocalWindow(cached.window_start_ms, cached.window_end_ms), cached.frames, best, cached.frames, cached.budget_exhausted, True)
            return self._score_sample(request, sampled, started)
        try:
            sampled = sample_two_stage(
                decoder=self._decoder, video_path=request.video_path,
                center_ms=request.candidate.timestamp_ms, radius_ms=radius,
                coarse_sample_fps=self._config.coarse_sample_fps,
                dense_radius_ms=self._config.dense_radius_ms,
                dense_sample_fps=self._config.dense_sample_fps,
                rank_frame=lambda frame: -abs(frame.timestamp_ms - request.candidate.timestamp_ms),
                budget=request.decode_budget, dense_seed_count=self._config.dense_seed_count,
            )
        except DecodeBudgetExhausted:
            # No original local frame was decoded, so TV5 cannot safely offer
            # a frame-step/manual selection. The caller still has its input
            # coarse candidate, but WP09 must make the unavailable state clear.
            raise RefinementUnavailable("decode_budget_exhausted")
        except RefinementUnavailable:
            raise
        except Exception as exc:
            raise RefinementUnavailable("decode_failure") from exc

        sampled = self._cache_sample(request, radius, sampled)
        return self._score_sample(request, sampled, started)

    def _score_sample(self, request: RefineRequest, sampled: SamplingResult, started: float) -> RefineResult:
        if self._scorer is None:
            return self._manual_result(request, sampled, "scorer_unavailable", started)
        try:
            scores = self._scorer.score(request.refinement_text, sampled.dense_frames)
            if len(scores) != len(sampled.dense_frames):
                raise ScoringUnavailable("scorer_unavailable")
        except ScoringUnavailable as exc:
            return self._manual_result(request, sampled, exc.reason, started)
        except Exception:
            return self._manual_result(request, sampled, "scorer_unavailable", started)

        frames = tuple(
            ScoredFrame(frame=frame, visual_score=float(score), evidence=tuple(item for item in request.evidence if item.frame_id in (None, frame.frame_id)))
            for frame, score in zip(sampled.dense_frames, scores)
        )
        selected = select_hypotheses(
            request.task, request.policy, frames, limit=self._config.hypothesis_limit,
            stable_variance_penalty=self._config.stable_variance_penalty,
        )
        hypotheses = tuple(self._automatic_hypothesis(request, sampled, item) for item in selected)
        status = RefinementStatus.PARTIAL if sampled.budget_exhausted else RefinementStatus.REFINED
        reason = "decode_budget_exhausted" if sampled.budget_exhausted else None
        return self._result(request, sampled, hypotheses, status, reason, started)

    def _request_cache_key(self, request: RefineRequest, radius: int) -> WindowRequestKey:
        budget = request.decode_budget
        return WindowRequestKey(request.context.preprocess_run_id, request.candidate.video_id, request.candidate.timestamp_ms, radius, (budget.max_decoded_frames, budget.max_window_ms, budget.max_decode_time_ms, budget.max_dense_regions), self._config.decoder_config)

    def _cache_sample(self, request: RefineRequest, radius: int, sampled: SamplingResult) -> SamplingResult:
        # PTS bounds come from resolved original frames, never timestamp/FPS estimates.
        frames = sampled.dense_frames
        key = DecodedWindowKey(request.context.preprocess_run_id, request.candidate.video_id, min(item.pts for item in frames), max(item.pts for item in frames), self._config.decoder_config)
        entry = CachedWindow(key, frames, sampled.window.start_ms, sampled.window.end_ms, sampled.best_coarse_frame.frame_id, sampled.budget_exhausted)
        self._cache.put_for_request(self._request_cache_key(request, radius), entry)
        return sampled

    def _automatic_hypothesis(self, request: RefineRequest, sampled: SamplingResult, item: ScoredFrame) -> ExactFrameHypothesis:
        return ExactFrameHypothesis(
            request.candidate.video_id, item.frame.frame_id, item.frame.timestamp_ms, item.policy_score,
            request.policy.value, visual_score=item.visual_score, policy_score=item.policy_score,
            window_start_ms=sampled.window.start_ms, window_end_ms=sampled.window.end_ms, evidence=item.evidence,
        )

    def _manual_result(self, request: RefineRequest, sampled: SamplingResult, reason: str, started: float) -> RefineResult:
        ordered = sorted(sampled.dense_frames, key=lambda frame: (abs(frame.timestamp_ms - request.candidate.timestamp_ms), frame.timestamp_ms, frame.frame_id))
        hypotheses = [
            ExactFrameHypothesis(request.candidate.video_id, frame.frame_id, frame.timestamp_ms, None, "manual_only", window_start_ms=sampled.window.start_ms, window_end_ms=sampled.window.end_ms)
            for frame in ordered[: self._config.hypothesis_limit]
        ]
        # Always preserve the upstream coarse frame exactly once for TV5 selection.
        if request.candidate.frame_id not in {item.frame_id for item in hypotheses}:
            hypotheses.insert(0, ExactFrameHypothesis(request.candidate.video_id, request.candidate.frame_id, request.candidate.timestamp_ms, request.candidate.upstream_score, "coarse_fallback", window_start_ms=sampled.window.start_ms, window_end_ms=sampled.window.end_ms))
        return self._result(request, sampled, tuple(hypotheses[: self._config.hypothesis_limit]), RefinementStatus.MANUAL_ONLY, reason, started)

    def _result(self, request: RefineRequest, sampled: SamplingResult, hypotheses: tuple[ExactFrameHypothesis, ...], status: RefinementStatus, reason: str | None, started: float) -> RefineResult:
        first = hypotheses[0] if hypotheses else None
        model_version = self._scorer.model_version if self._scorer is not None else request.context.model_version
        audit = RefinementAudit(
            context=request.context, before_frame_id=request.candidate.frame_id,
            before_score=request.candidate.upstream_score, after_frame_id=first.frame_id if first else None,
            after_score=first.score if first else None, window_start_ms=sampled.window.start_ms,
            window_end_ms=sampled.window.end_ms, model_version=model_version,
            config_version=self._config.config_version, latency_ms=(monotonic() - started) * 1_000,
            cache_hit=sampled.cache_hit, decoded_frame_count=len(sampled.dense_frames),
        )
        modalities = {"visual": "available" if self._scorer is not None and status is not RefinementStatus.MANUAL_ONLY else "unavailable", "evidence": "available" if request.evidence else "unavailable"}
        return RefineResult(request.candidate, hypotheses, reason, status, request.context, audit, modalities)
