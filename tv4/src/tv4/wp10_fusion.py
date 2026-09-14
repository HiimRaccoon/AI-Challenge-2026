"""WP10 — Multimodal Fusion (RRF) and the KIS orchestration workflow.

Combines independent branch results (visual from TV2/WP03, ocr/asr/object/
metadata from TV3) into one ranked, deduplicated, video-diverse top-100 list
of `SearchCandidate`, tolerant of any single branch being empty/unavailable
(object is always a *soft* signal, never a hard filter, per the plan).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .contracts import SearchCandidate

DEFAULT_RRF_K = 60
DEFAULT_DEDUP_WINDOW_MS = 5_000

# Branches TV4 treats as soft/boost-only signals rather than hard filters —
# a candidate is never *dropped* purely for missing an object hint.
SOFT_BRANCHES = {"object", "metadata"}

# Reliability priors apply before rank fusion. OCR, generic object labels and
# metadata are useful corroboration, but a one-branch rank-1 false positive
# must not tie a rank-1 visual match merely because both happen to be first in
# indexes with very different score distributions.
BRANCH_RELIABILITY = {
    "visual": 1.00,
    "ocr": 0.75,
    "asr": 1.00,
    "object": 0.35,
    "metadata": 0.25,
}


def _candidate_quality(branch: str, candidate: SearchCandidate) -> float:
    """Calibrate bounded semantic ASR scores without comparing score families.

    Visual RRF, BM25 and detector confidence use incompatible scales. Only
    semantic ASR cosine is recognized here (0..1); weak nearest-neighbor hits
    are downweighted rather than receiving the same vote as a strong match.
    """
    score = candidate.raw_score
    reliability = BRANCH_RELIABILITY.get(branch, 1.0)
    if branch != "asr" or isinstance(score, bool) or not isinstance(score, (int, float)):
        return reliability
    score = float(score)
    if not 0.0 <= score <= 1.0:
        return reliability
    # Semantic nearest-neighbour lists always have a rank 1, even when the
    # corpus contains no good transcript match. Scores below roughly 0.52 are
    # background similarity; ramp them only through the credible 0.52..0.70
    # range instead of treating a generic 0.54 hit as medium confidence.
    calibrated = max(0.0, min(1.0, (score - 0.52) / 0.18))
    # Keep a very small vote for uncertain transcript evidence so it remains
    # inspectable, but do not let a weak nearest neighbour promote an
    # otherwise lower-ranked visual lookalike above the best visual result.
    return reliability * (0.05 + 0.95 * calibrated)


def reciprocal_rank_fusion(
    branch_results: dict[str, list[SearchCandidate]],
    *,
    weights: dict[str, float] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    dedup_window_ms: int = DEFAULT_DEDUP_WINDOW_MS,
    top_k: int = 100,
    min_relative_score: float | None = None,
) -> list[SearchCandidate]:
    """Standard RRF: score(c) = sum_over_branches weight_m * quality_m / (rrf_k + rank).

    Candidates from the same video are clustered around the strongest item
    when their timestamps differ by at most `dedup_window_ms`. A real
    distance check avoids fixed-bucket boundary errors (for example, 4.9 s
    and 5.1 s are close even though integer buckets say otherwise). Within a
    cluster, each branch gets at most one vote so dense keyframe sampling
    cannot make one modality dominate fusion.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if dedup_window_ms <= 0:
        raise ValueError("dedup_window_ms must be positive")
    if min_relative_score is not None and not 0 < min_relative_score <= 1:
        raise ValueError("min_relative_score must be in (0, 1]")

    by_video: dict[str, list[tuple[str, SearchCandidate, float]]] = defaultdict(list)
    for branch, candidates in branch_results.items():
        w = float(weights.get(branch, 1.0)) if weights and branch in weights else 1.0
        if w <= 0.0:
            continue
        for c in candidates:
            contribution = w * _candidate_quality(branch, c) / (rrf_k + c.rank)
            by_video[c.video_id].append((branch, c, contribution))

    buckets: list[dict] = []
    for items in by_video.values():
        # Strong anchors are established first. This makes clustering stable
        # around useful evidence instead of depending on branch iteration order.
        items.sort(key=lambda item: (-item[2], item[1].timestamp_ms))
        video_buckets: list[dict] = []
        for branch, candidate, contribution in items:
            close = [
                entry for entry in video_buckets
                if abs(candidate.timestamp_ms - entry["anchor_timestamp_ms"]) <= dedup_window_ms
            ]
            if close:
                entry = min(
                    close,
                    key=lambda value: abs(candidate.timestamp_ms - value["anchor_timestamp_ms"]),
                )
            else:
                entry = {
                    "candidate": candidate,
                    "anchor_timestamp_ms": candidate.timestamp_ms,
                    "representative_contribution": contribution,
                    "branch_contributions": {},
                    "branch_provenance": {},
                    "model_scores": {},
                }
                video_buckets.append(entry)

            previous = entry["branch_contributions"].get(branch, -1.0)
            if contribution > previous:
                entry["branch_contributions"][branch] = contribution
                entry["branch_provenance"][branch] = dict(candidate.provenance)
                if candidate.score is not None:
                    entry["model_scores"][branch] = candidate.score
            if contribution > entry["representative_contribution"]:
                entry["candidate"] = candidate
                entry["representative_contribution"] = contribution
        buckets.extend(video_buckets)

    fused: list[SearchCandidate] = []
    for entry in buckets:
        base = entry["candidate"]
        visual_provenance = entry["branch_provenance"].get("visual", {})
        fused.append(
            replace(
                base,
                source="fusion",
                score=sum(entry["branch_contributions"].values()),
                provenance_sources=tuple(sorted(entry["branch_contributions"])),
                model_scores={**base.model_scores, **entry["model_scores"]},
                provenance={
                    **base.provenance,
                    "branch_provenance": entry["branch_provenance"],
                    **{
                        key: visual_provenance[key]
                        for key in ("visual_query_variants", "visual_video_support_count")
                        if key in visual_provenance
                    },
                },
            )
        )

    fused.sort(key=lambda c: c.score or 0.0, reverse=True)
    # Relative pruning is opt-in. A high-confidence false positive must not
    # erase lower-ranked videos before QA/TRAKE verification can inspect them.
    if fused and min_relative_score is not None:
        score_floor = (fused[0].score or 0.0) * min_relative_score
        fused = [candidate for candidate in fused if (candidate.score or 0.0) >= score_floor]
    return _diversify_top_k(fused, top_k=top_k)


def _diversify_top_k(ranked: list[SearchCandidate], *, top_k: int, per_video_cap: int = 5) -> list[SearchCandidate]:
    """Cap how many frames per single video can occupy the top slots.

    Prevents one very-well-matched video from filling the whole top-100 and
    starving the "top-20 diversity" gate the plan calls for (G1/G3). This cap
    is enforced strictly -- a query genuinely matching only one video still
    returns at most `per_video_cap` frames rather than padding out to
    `top_k` with more near-duplicates of the same video.
    """
    per_video_count: dict[str, int] = defaultdict(int)
    kept: list[SearchCandidate] = []
    for c in ranked:
        if per_video_count[c.video_id] < per_video_cap:
            kept.append(c)
            per_video_count[c.video_id] += 1
        if len(kept) >= top_k:
            break
    return [replace(c, rank=i + 1) for i, c in enumerate(kept[:top_k])]


def fuse_kis(
    branch_results: dict[str, list[SearchCandidate]],
    *,
    weights: dict[str, float] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    dedup_window_ms: int = DEFAULT_DEDUP_WINDOW_MS,
    top_k: int = 100,
    min_relative_score: float | None = None,
) -> list[SearchCandidate]:
    """KIS/WP10 entry point. Object/metadata contribute score only, never a hard gate."""
    return reciprocal_rank_fusion(
        branch_results, weights=weights, rrf_k=rrf_k, dedup_window_ms=dedup_window_ms,
        top_k=top_k, min_relative_score=min_relative_score,
    )
