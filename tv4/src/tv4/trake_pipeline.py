"""End-to-end TRAKE orchestration: WP07 event split -> WP10 fusion (whole
query, for video retrieval) -> per-event WP10 fusion (for alignment) -> WP12
temporal alignment -> optional WP09 fine-alignment per event.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Sequence

from .contracts import SearchCandidate, SearchRequest, TrakeHypothesis
from .kis_pipeline import KisServices, canonical_refined_candidate
from .media_identity import resolve_original_media_path
from .wp07_router import route_trake
from .wp10_fusion import fuse_kis
from .wp12_trake import EventPool, align_trake, build_event_pools, to_hypothesis
from .trake_kbest import align_k_best, event_alternatives
from .trake_verifier import TrakeVerification, verify_sequence
from .visual_verifier import (
    OllamaVisualVerifier,
    canonical_trake_frame_groups,
)
from .query_enhancer import enhance_visual_query
from .trake_event_parser import parse_event_graph
from .external_assist import ExternalQueryAssist


def visual_query_for_trake(query_text: str, *, context: str | None = None) -> str:
    """Build one domain-independent English-oriented event query.

    Short TRAKE events often omit their subject, so the sequence description
    is retained as context. Concept expansion is shared with KIS instead of
    embedding benchmark-topic rules here. Online translation is deliberately
    disabled on this latency-critical path.
    """
    event_query = enhance_visual_query(query_text, allow_online_translation=False)
    if not context or not context.strip():
        return event_query
    context_query = enhance_visual_query(context, allow_online_translation=False)
    if context_query.casefold() == event_query.casefold():
        return event_query
    return f"{event_query}; sequence context: {context_query}"


def _align_visual_sequence(
    whole_visual: list[SearchCandidate],
    per_event_visual: list[list[SearchCandidate]],
    *,
    strategy: str,
) -> tuple[str, list[SearchCandidate]] | None:
    """Choose the visual video that supports the complete ordered sequence.

    The previous top-1 policy selected a video before checking whether any
    event was present in it. Evaluate every whole-query visual hypothesis;
    candidates without all event pools are discarded and the best monotonic
    visual path wins. This prevents a semantically unrelated top-1 frame
    from being copied into every TRAKE slot.
    """
    best: tuple[tuple[float, int, str], str, list[SearchCandidate]] | None = None
    # A full-prompt visual query may describe several actions and rank the
    # correct video below its top-k cutoff, while every individual event still
    # retrieves it. Consider the union of full-prompt and event hypotheses;
    # the full-query rank is only a deterministic tie-breaker after complete
    # ordered event coverage has been proven.
    whole_rank: dict[str, int] = {}
    video_order: list[str] = []
    for candidate in whole_visual:
        if candidate.video_id not in whole_rank:
            whole_rank[candidate.video_id] = candidate.rank
            video_order.append(candidate.video_id)
    for event_candidates in per_event_visual:
        for candidate in event_candidates:
            if candidate.video_id not in whole_rank:
                whole_rank[candidate.video_id] = len(whole_visual) + 1
                video_order.append(candidate.video_id)
    for video_id in video_order:
        pools = build_event_pools(video_id, per_event_visual)
        aligned = align_trake(pools, strategy=strategy)
        if aligned is None:
            continue
        # Event evidence is the primary objective. Use whole-query rank only
        # as deterministic tie-breaker, never as a substitute for coverage.
        key = (sum(candidate.score or 0.0 for candidate in aligned), -whole_rank[video_id], video_id)
        if best is None or key > best[0]:
            best = (key, video_id, aligned)
    return None if best is None else (best[1], best[2])


def _align_visual_kbest(
    whole_visual: list[SearchCandidate],
    per_event_visual: list[list[SearchCandidate]],
    *,
    strategy: str,
    limit: int,
) -> list[tuple[str, list[SearchCandidate], list[EventPool]]]:
    """Extend the proven Top-1 alignment with bounded alternative paths.

    The first item is forced to the existing alignment result so enabling
    Top-K cannot change the legacy ``result`` consumed by TV5.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    legacy = _align_visual_sequence(whole_visual, per_event_visual, strategy=strategy)
    if legacy is None:
        return []

    whole_rank: dict[str, int] = {}
    video_order: list[str] = []
    for candidate in whole_visual:
        if candidate.video_id not in whole_rank:
            whole_rank[candidate.video_id] = candidate.rank
            video_order.append(candidate.video_id)
    for event_candidates in per_event_visual:
        for candidate in event_candidates:
            if candidate.video_id not in whole_rank:
                whole_rank[candidate.video_id] = len(whole_visual) + 1
                video_order.append(candidate.video_id)

    raw: list[tuple[tuple[float, int, str], str, list[SearchCandidate], list[EventPool]]] = []
    for video_id in video_order:
        pools = build_event_pools(video_id, per_event_visual)
        for path in align_k_best(pools, k=limit):
            raw.append((
                (path.score, -whole_rank[video_id], video_id),
                video_id,
                list(path.candidates),
                list(pools),
            ))
    raw.sort(key=lambda item: item[0], reverse=True)

    legacy_video, legacy_path = legacy
    legacy_identity = tuple((item.video_id, item.frame_id) for item in legacy_path)
    legacy_pools = list(build_event_pools(legacy_video, per_event_visual))
    selected: list[tuple[str, list[SearchCandidate], list[EventPool]]] = [
        (legacy_video, legacy_path, legacy_pools),
    ]
    seen = {legacy_identity}
    for _, video_id, path, pools in raw:
        identity = tuple((item.video_id, item.frame_id) for item in path)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append((video_id, path, pools))
        if len(selected) == limit:
            break
    return selected


def _refine_trake_alignment(
    aligned: list[SearchCandidate], events: tuple[str, ...], services: KisServices,
) -> list[SearchCandidate]:
    """Tighten already aligned event anchors against original frames.

    Retrieval keyframes are intentionally sparse and may be seconds from the
    requested action. WP09 is the only component allowed to replace them
    because it returns a certified original-frame identity. Keep the coarse
    path whenever refinement is unavailable or would violate event ordering.
    """
    if services.refine is None or services.original_media_root is None:
        return aligned
    if len(aligned) != len(events):
        return aligned
    try:
        media_path = resolve_original_media_path(
            services.original_media_root, aligned[0].video_id, services.media_registry,
            services.preprocess_run_id, services.allowed_media_extensions,
        )
    except ValueError:
        return aligned

    refined: list[SearchCandidate] = []
    for candidate, event_text in zip(aligned, events):
        try:
            result = services.refine.refine(
                {
                    "candidate": {
                        "video_id": candidate.video_id,
                        "frame_id": candidate.frame_id,
                        "timestamp_ms": candidate.timestamp_ms,
                        "upstream_score": candidate.score,
                        "confidence": candidate.confidence,
                    },
                    "video_path": str(media_path),
                    "task": "TRAKE",
                    "refinement_text": event_text,
                    "policy": "peak",
                    "context": {
                        "preprocess_run_id": services.preprocess_run_id,
                        "media_record_ref": candidate.video_id,
                        "mapping_ref": f"tv1-frames/{candidate.video_id}",
                        "decoder_version": "pyav-tv4-adapter-1",
                        "model_version": "n/a",
                        "config_version": "default",
                    },
                    "decode_budget": {
                        "max_decoded_frames": 64,
                        "max_window_ms": 4_000,
                        "max_decode_time_ms": 8_000,
                        "max_dense_regions": 3,
                    },
                }
            )
        except Exception:
            return aligned
        hypotheses = result.get("hypotheses") if isinstance(result, dict) else None
        best = canonical_refined_candidate(hypotheses[0]) if isinstance(hypotheses, list) and hypotheses else None
        if best is None:
            return aligned
        refined.append(replace(candidate, frame_id=best[0], timestamp_ms=best[1]))

    if any(
        later.video_id != earlier.video_id or later.timestamp_ms <= earlier.timestamp_ms
        for earlier, later in zip(refined, refined[1:])
    ):
        return aligned
    return refined


def _fuse_for_request(request: SearchRequest, services: KisServices, query_text: str) -> list[SearchCandidate]:
    branch_results: dict[str, list[SearchCandidate]] = {
        "visual": services.visual.search(request.query_id, query_text, event_index=request.event_index),
    }
    tv3_results = services.tv3.search_all(request, ["ocr", "asr", "object"])
    branch_results.update(tv3_results)
    normalized = {
        branch: [replace(candidate, rank=index) for index, candidate in enumerate(candidates, start=1)]
        for branch, candidates in branch_results.items()
    }
    return fuse_kis(normalized, top_k=request.limit)


def _fuse_event_candidates(
    request: SearchRequest,
    visual_candidates: list[SearchCandidate],
    services: KisServices,
) -> list[SearchCandidate]:
    """Fuse per-event modalities without changing the shared KIS pipeline."""
    branch_results: dict[str, list[SearchCandidate]] = {"visual": visual_candidates}
    branch_results.update(services.tv3.search_all(request, ["ocr", "asr", "object"]))
    normalized = {
        branch: [replace(candidate, rank=index) for index, candidate in enumerate(candidates, start=1)]
        for branch, candidates in branch_results.items()
    }
    return fuse_kis(normalized, top_k=request.limit)


def _locked_bounds(
    event_index: int,
    locked_events: Sequence[tuple[int, int]],
) -> tuple[int | None, int | None]:
    lower_values = [timestamp for index, timestamp in locked_events if index < event_index]
    upper_values = [timestamp for index, timestamp in locked_events if index > event_index]
    return (
        max(lower_values) if lower_values else None,
        min(upper_values) if upper_values else None,
    )


def rerun_trake_event(
    query_text: str,
    events: Sequence[str],
    event_index: int,
    video_id: str,
    services: KisServices,
    *,
    locked_events: Sequence[tuple[int, int]] = (),
    query_id: str | None = None,
    exclude_frame_id: int | None = None,
    limit: int = 5,
) -> tuple[list[SearchCandidate], int | None, int | None]:
    """Retrieve one event while treating locked slots as immutable bounds."""
    if not 0 <= event_index < len(events):
        raise ValueError("event_index is outside events")
    if not video_id.strip():
        raise ValueError("video_id is required")
    if not 1 <= limit <= 5:
        raise ValueError("limit must be in [1, 5]")
    locked_indexes = [index for index, _ in locked_events]
    if len(locked_indexes) != len(set(locked_indexes)):
        raise ValueError("locked event indexes must be unique")
    if any(index < 0 or index >= len(events) for index in locked_indexes):
        raise ValueError("locked event index is outside events")
    if event_index in locked_indexes:
        raise ValueError("target event must be unlocked before rerun")
    ordered_locks = sorted(locked_events)
    if any(left[1] >= right[1] for left, right in zip(ordered_locks, ordered_locks[1:])):
        raise ValueError("locked event timestamps must be strictly increasing")

    decision = route_trake(query_text, events=list(events), query_id=query_id)
    root_request = decision.request
    event_text = root_request.events[event_index]
    event_request = replace(
        root_request,
        query_id=f"{root_request.query_id}-rerun-ev{event_index}",
        query_text=event_text,
        event_index=event_index,
        events=(),
    )
    visual_query = visual_query_for_trake(event_text, context=query_text)
    candidates = _fuse_for_request(event_request, services, visual_query)
    lower, upper = _locked_bounds(event_index, locked_events)
    filtered = [
        candidate for candidate in candidates
        if candidate.video_id == video_id
        and candidate.frame_id != exclude_frame_id
        and (lower is None or candidate.timestamp_ms > lower)
        and (upper is None or candidate.timestamp_ms < upper)
    ]
    deduplicated: list[SearchCandidate] = []
    seen: set[int] = set()
    for candidate in filtered:
        if candidate.frame_id in seen:
            continue
        seen.add(candidate.frame_id)
        deduplicated.append(replace(
            candidate,
            query_id=root_request.query_id,
            event_index=event_index,
            preprocess_run_id=services.preprocess_run_id,
        ))
        if len(deduplicated) == limit:
            break
    return deduplicated, lower, upper


def _event_coverage(whole, per_event):
    video_ranks = {}
    for candidate in whole:
        video_ranks.setdefault(candidate.video_id, candidate.rank)
    for pool in per_event:
        for candidate in pool:
            video_ranks.setdefault(candidate.video_id, len(whole) + 1)
    rows = []
    for video_id, rank in video_ranks.items():
        counts = [sum(c.video_id == video_id for c in pool) for pool in per_event]
        rows.append({
            "video_id": video_id, "whole_video_rank": rank,
            "event_candidate_counts": counts,
            "missing_event_indexes": [i for i, count in enumerate(counts) if not count],
        })
    return sorted(rows, key=lambda row: (len(row["missing_event_indexes"]), row["whole_video_rank"], row["video_id"]))[:5]


def _supplement_event_candidates(services, visual_queries, per_event, coverage):
    """One scoped batch for incomplete sequences; all identities come from TV1."""
    scope = []
    for row in coverage[:3]:
        video_id = row["video_id"]
        frames = services.tv1.frames(video_id)
        frames = [f for f in frames if isinstance(f, dict)
                  and f.get("video_id") == video_id
                  and f.get("preprocess_run_id") == services.preprocess_run_id
                  and isinstance(f.get("frame_id"), int) and not isinstance(f.get("frame_id"), bool)
                  and f["frame_id"] >= 0]
        if len(frames) > 500:
            frames = [frames[i * (len(frames) - 1) // 499] for i in range(500)]
        scope.extend({"video_id": video_id, "frame_id": f["frame_id"]} for f in frames)
    if not scope or not hasattr(services.visual, "search_batch"):
        return per_event, False
    extra = services.visual.search_batch(visual_queries[1:], candidate_frames=scope, top_k=100)
    if len(extra) != len(per_event):
        raise ValueError("supplemental event batch length mismatch")
    merged = []
    for original, additional in zip(per_event, extra):
        by_id = {}
        for candidate in original + additional:
            identity = (candidate.video_id, candidate.frame_id)
            previous = by_id.get(identity)
            if previous is None or (candidate.score or 0.0) > (previous.score or 0.0):
                by_id[identity] = candidate
        merged.append(sorted(by_id.values(), key=lambda c: (-(c.score or 0.0), c.rank, c.video_id, c.frame_id))[:200])
    return merged, True


def run_trake_queries(
    query_text: str,
    services: KisServices,
    *,
    events: list[str] | None = None,
    query_id: str | None = None,
    strategy: str = "dp",
    limit: int = 5,
    visual_verifier: OllamaVisualVerifier | None = None,
    diagnostics: dict | None = None,
) -> list[TrakeHypothesis]:
    decision = route_trake(query_text, events=events, query_id=query_id)
    request = decision.request

    # Build requests for whole query (idx 0) and all ordered events (idx 1..N)
    batch_specs: list[tuple[str, str, int | None, SearchRequest]] = []
    whole_req = replace(request, events=())
    batch_specs.append((request.query_id, query_text, None, whole_req))

    for idx, event_text in enumerate(request.events):
        event_req = replace(
            request,
            query_id=f"{request.query_id}-ev{idx}",
            query_text=event_text,
            event_index=idx,
            events=(),
        )
        batch_specs.append((event_req.query_id, event_text, idx, event_req))

    # Single batched visual search across whole sequence and all events
    visual_queries = [
        (qid, visual_query_for_trake(qtxt, context=query_text if ev_idx is not None else None), ev_idx)
        for qid, qtxt, ev_idx, _ in batch_specs
    ]
    translations = ExternalQueryAssist.from_environment().trake_translations(query_text, tuple(request.events))
    visual_queries = [
        (qid, f"{translation}\nOriginal event and context: {text}" if translation else text, index)
        for (qid, text, index), translation in zip(visual_queries, translations)
    ]
    if hasattr(services.visual, "search_batch"):
        visual_results = services.visual.search_batch(visual_queries)
    else:
        visual_results = [
            services.visual.search(qid, qtxt, event_index=ev_idx)
            for qid, qtxt, ev_idx in visual_queries
        ]

    # Stage 1 is a visual identification task. Text branches can add useful
    # evidence in other surfaces, but must not override an actual visual hit
    # with unrelated lexical overlap.
    whole_visual = visual_results[0]
    event_specs = batch_specs[1:]
    if not event_specs:
        return []
    with ThreadPoolExecutor(
        max_workers=min(4, len(event_specs)), thread_name_prefix="trake-multimodal",
    ) as executor:
        fused_per_event = list(executor.map(
            lambda pair: _fuse_event_candidates(pair[0][3], pair[1], services),
            zip(event_specs, visual_results[1:]),
        ))
    # A full-sequence embedding can miss the correct video even when every
    # individual event retrieves it. Alignment already evaluates the bounded
    # union of whole-query and per-event candidates, so the noisy whole-query
    # list must not become a hard video filter.
    per_event_candidates = fused_per_event
    raw_limit = max(10, limit * 2)
    selected = _align_visual_kbest(
        whole_visual, per_event_candidates, strategy=strategy, limit=raw_limit,
    )
    coverage = _event_coverage(whole_visual, per_event_candidates)
    details = diagnostics if diagnostics is not None else {}
    details.update({
        "event_candidate_counts": [len(pool) for pool in per_event_candidates],
        "video_coverage_before": coverage,
        "supplement_attempted": False,
        "events": [{"event_index": e.event_index, "actions": list(e.actions),
                    "temporal_cues": list(e.temporal_cues), "ordinal": e.ordinal}
                   for e in parse_event_graph(request.events)],
    })
    if not selected and coverage:
        try:
            per_event_candidates, attempted = _supplement_event_candidates(
                services, visual_queries, per_event_candidates, coverage,
            )
            details["supplement_attempted"] = attempted
            if attempted:
                selected = _align_visual_kbest(whole_visual, per_event_candidates, strategy=strategy, limit=raw_limit)
        except Exception as exc:
            details["supplement_error"] = type(exc).__name__
    details["video_coverage_after"] = _event_coverage(whole_visual, per_event_candidates)
    details["alignment_found"] = bool(selected)
    whole_video_rank: dict[str, int] = {}
    for candidate in whole_visual:
        whole_video_rank.setdefault(candidate.video_id, candidate.rank)
    verified_paths: list[
        tuple[TrakeVerification, str, list[SearchCandidate], list[EventPool]]
    ] = []
    for video_id, raw_aligned, raw_pools in selected:
        verification = verify_sequence(
            tuple(request.events),
            raw_aligned,
            whole_video_rank=whole_video_rank.get(video_id, len(whole_visual) + 1),
        )
        verified_paths.append((verification, video_id, raw_aligned, raw_pools))
    verified_paths.sort(
        key=lambda item: (
            -item[0].final_sequence_score,
            -sum(candidate.score or 0.0 for candidate in item[2]),
            item[1],
            tuple(candidate.frame_id for candidate in item[2]),
        )
    )
    # The VLM is an optional late verifier, never a retrieval source.  Only a
    # bounded number of already-ranked sequence candidates reach it, and all
    # image identities/paths come from TV1's canonical frame catalogue.
    verifier = visual_verifier or OllamaVisualVerifier.from_environment()
    if verifier.enabled and services.derivative_asset_root is not None:
        checked: list[
            tuple[TrakeVerification, str, list[SearchCandidate], list[EventPool]]
        ] = []
        for index, (verification, video_id, raw_aligned, raw_pools) in enumerate(verified_paths):
            if index < verifier.max_sequence_candidates:
                frame_groups = canonical_trake_frame_groups(
                    raw_aligned,
                    services.tv1,
                    services.derivative_asset_root,
                    services.allowed_image_extensions,
                    preprocess_run_id=services.preprocess_run_id,
                    limit_per_event=verifier.max_images,
                )
                visual_checks = verifier.verify_events(tuple(request.events), frame_groups)
                verification = verify_sequence(
                    tuple(request.events),
                    raw_aligned,
                    whole_video_rank=whole_video_rank.get(video_id, len(whole_visual) + 1),
                    visual_verifications=visual_checks,
                )
            checked.append((verification, video_id, raw_aligned, raw_pools))
        verified_paths = checked
        verified_paths.sort(
            key=lambda item: (
                -item[0].final_sequence_score,
                -sum(candidate.score or 0.0 for candidate in item[2]),
                item[1],
                tuple(candidate.frame_id for candidate in item[2]),
            )
        )
    hypotheses: list[TrakeHypothesis] = []
    for rank, (verification, video_id, raw_aligned, raw_pools) in enumerate(
        verified_paths[:limit], start=1,
    ):
        aligned = [
            replace(candidate, query_id=request.query_id, preprocess_run_id=services.preprocess_run_id)
            for candidate in raw_aligned
        ]
        # Preserve the established, potentially expensive WP09 refinement for
        # the legacy Top-1 only. Alternatives remain producer-owned keyframes.
        if rank == 1:
            aligned = _refine_trake_alignment(aligned, tuple(request.events), services)
            refined_visual_checks = None
            if verifier.enabled and services.derivative_asset_root is not None:
                refined_groups = canonical_trake_frame_groups(
                    aligned,
                    services.tv1,
                    services.derivative_asset_root,
                    services.allowed_image_extensions,
                    preprocess_run_id=services.preprocess_run_id,
                    limit_per_event=verifier.max_images,
                )
                refined_visual_checks = verifier.verify_events(
                    tuple(request.events), refined_groups,
                )
            verification = verify_sequence(
                tuple(request.events),
                aligned,
                whole_video_rank=whole_video_rank.get(video_id, len(whole_visual) + 1),
                visual_verifications=refined_visual_checks,
            )
        alternatives = event_alternatives(raw_pools, aligned, limit=5)
        alternatives = tuple(
            tuple(
                replace(candidate, query_id=request.query_id, preprocess_run_id=services.preprocess_run_id)
                for candidate in event_candidates
            )
            for event_candidates in alternatives
        )
        hypotheses.append(to_hypothesis(
            request.query_id,
            video_id,
            aligned,
            services.preprocess_run_id,
            rank=rank,
            event_alternatives=alternatives,
            verification=verification.to_json(),
        ))
    return hypotheses


def run_trake_query(
    query_text: str,
    services: KisServices,
    *,
    events: list[str] | None = None,
    query_id: str | None = None,
    strategy: str = "dp",
    visual_verifier: OllamaVisualVerifier | None = None,
) -> TrakeHypothesis | None:
    """Backward-compatible single-result wrapper over the K-best pipeline."""
    hypotheses = run_trake_queries(
        query_text,
        services,
        events=events,
        query_id=query_id,
        strategy=strategy,
        limit=5,
        visual_verifier=visual_verifier,
    )
    return hypotheses[0] if hypotheses else None
