"""End-to-end Textual-KIS orchestration: router (WP07) -> retrievers -> fusion (WP10)
-> optional WP09 exact-frame refinement -> submission-ready top-100.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import defaultdict
from pathlib import Path
import re
import unicodedata
from typing import Any, Literal, Mapping
from concurrent.futures import ThreadPoolExecutor

from .clients.tv1_client import TV1Client
from .clients.tv2_refine_client import TV2RefineClient
from .clients.tv2_visual_client import TV2VisualClient
from .clients.tv3_client import TV3Client
from .contracts import SearchCandidate
from .wp07_router import route_kis, route_qa
from .wp10_fusion import fuse_kis
from .external_assist import ExternalVlmVerifier, is_temporal_visual_query, split_visual_moments
from .qa_query_plan import build_qa_query_plan, question_prefers_visual_text
from .query_enhancer import semantic_text_locator_query


_ASR_ANSWER_CUES = re.compile(
    r"\b("
    r"o\s+dau|where|khi\s+nao|bao\s+gio|when|ai|who|"
    r"tai\s+sao|vi\s+sao|why|"
    r"ten\s+la\s+gi|xa\s+nao|dia\s+diem|duong\s+nao|ten\s+duong|"
    r"nam\s+tren\s+duong|what\s+is\s+the\s+name"
    r")\b",
    re.IGNORECASE,
)

_ASR_SCOPE_MIN_TOP_SCORE = 0.58
_ASR_SCOPE_MIN_WINDOW_SCORE = 0.52
_ASR_SCOPE_MAX_DROP_FROM_TOP = 0.12


def _question_prefers_asr(question: str | None, description: str = "") -> bool:
    """Whether a VQA answer normally needs spoken/textual temporal context."""
    if not question or question_prefers_visual_text(question, description):
        return False
    folded = "".join(
        char for char in unicodedata.normalize("NFD", question.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")
    return bool(_ASR_ANSWER_CUES.search(folded))


def _fold_text(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def _description_for_vqa_retrieval(query_text: str, question: str) -> str:
    """Remove a duplicated trailing question before lexical video retrieval.

    The UI exposes the event description and question as distinct fields, but
    imported contest prompts often place both in ``query_text``. Letting the
    question's generic words (for example, ``tên là gì``) into BM25 can rank
    unrelated educational videos above the named event. This only removes a
    sentence explicitly marked as a question; it never rewrites the event.
    """
    original = query_text.strip()
    if not original:
        return original
    question_folded = _fold_text(question)
    parts = re.split(r"(?<=[.!?])\s+", original)
    kept: list[str] = []
    for part in parts:
        folded_part = _fold_text(part)
        if folded_part.startswith(("hoi ", "cau hoi")) or (question_folded and question_folded in folded_part):
            # A sentence split after ``?`` can leave a parenthetical timing
            # qualifier as the next fragment. Stop at the question rather
            # than retaining that orphaned suffix as retrieval noise.
            break
        kept.append(part)
    result = " ".join(kept).strip()
    return result or original


def _transcript_frame_scope(
    services: "KisServices", *, query_id: str, query_text: str,
) -> tuple[list[dict[str, object]], list[SearchCandidate]]:
    """Return the bounded canonical frame scope supplied by TV3 ASR evidence.

    This is additive: unavailable/older TV3 deployments simply leave the
    visual search global. No timestamp-to-frame calculation occurs in TV4.
    """
    search_windows = getattr(services.tv3, "search_transcript_windows", None)
    if not callable(search_windows):
        return [], []
    try:
        windows = search_windows(query_id=query_id, query_text=query_text, limit=20, context_ms=10_000)
    except Exception:
        return [], []
    semantic_scores = [
        float(window["raw_score"])
        for window in windows
        if isinstance(window, dict)
        and isinstance(window.get("raw_score"), (int, float))
        and not isinstance(window.get("raw_score"), bool)
        and 0.0 <= float(window["raw_score"]) <= 1.0
    ]
    top_semantic_score = max(semantic_scores, default=0.0)
    scope_score_floor = max(
        _ASR_SCOPE_MIN_WINDOW_SCORE,
        top_semantic_score - _ASR_SCOPE_MAX_DROP_FROM_TOP,
    )
    allow_transcript_scope = top_semantic_score >= _ASR_SCOPE_MIN_TOP_SCORE
    scope: list[dict[str, object]] = []
    candidates: list[SearchCandidate] = []
    seen: set[tuple[str, int]] = set()
    for window in windows:
        video_id = window.get("video_id") if isinstance(window, dict) else None
        frames = window.get("canonical_frames", ()) if isinstance(window, dict) else ()
        if not isinstance(video_id, str) or not isinstance(frames, list):
            continue
        valid_frames = [
            frame for frame in frames
            if isinstance(frame, dict)
            and isinstance(frame.get("frame_id"), int) and not isinstance(frame.get("frame_id"), bool)
            and isinstance(frame.get("timestamp_ms"), int) and not isinstance(frame.get("timestamp_ms"), bool)
        ]
        start_ms, end_ms = window.get("start_ms"), window.get("end_ms")
        if valid_frames and isinstance(start_ms, int) and isinstance(end_ms, int):
            midpoint = (start_ms + end_ms) // 2
            anchor = min(valid_frames, key=lambda frame: (abs(int(frame["timestamp_ms"]) - midpoint), int(frame["timestamp_ms"]), int(frame["frame_id"])))
            candidates.append(SearchCandidate(
                query_id=query_id, video_id=video_id, frame_id=int(anchor["frame_id"]),
                timestamp_ms=int(anchor["timestamp_ms"]), source="asr", rank=len(candidates) + 1,
                window_start_ms=start_ms, window_end_ms=end_ms,
                raw_score=float(window.get("raw_score")) if isinstance(window.get("raw_score"), (int, float)) else None,
                evidence_refs=tuple(str(value) for value in window.get("evidence_refs", ()) if isinstance(value, str)),
                provenance_sources=("asr",), confidence=window.get("confidence"),
                preprocess_run_id=str(window.get("preprocess_run_id", "unknown")),
            ))
        window_score = window.get("raw_score")
        window_can_scope = (
            allow_transcript_scope
            and isinstance(window_score, (int, float))
            and not isinstance(window_score, bool)
            and float(window_score) >= scope_score_floor
        )
        for frame in (valid_frames if window_can_scope else ()):
            frame_id = frame.get("frame_id") if isinstance(frame, dict) else None
            if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                continue
            identity = (video_id, frame_id)
            if identity not in seen:
                seen.add(identity)
                scope.append({"video_id": video_id, "frame_id": frame_id})
            if len(scope) == 500:
                return scope, candidates
    return scope, candidates


def _frame_scope_around_text_anchors(
    services: "KisServices", anchors: list[SearchCandidate], *,
    window_ms: int = 30_000, max_videos: int = 8, max_frames: int = 1_500,
) -> list[dict[str, object]]:
    """Expand strong text anchors into a bounded TV1-owned visual scope.

    A transcript hit identifies a topic interval, not necessarily the exact
    frame on which its illustrative slide is fully revealed.  This keeps the
    global visual search intact while allowing WP03 to exactly rescore nearby
    canonical keyframes in the same videos.
    """
    if not anchors or window_ms <= 0 or max_videos <= 0 or max_frames <= 0:
        return []
    video_order: list[str] = []
    anchors_by_video: dict[str, list[int]] = defaultdict(list)
    for anchor in anchors:
        if anchor.video_id not in anchors_by_video:
            if len(video_order) >= max_videos:
                continue
            video_order.append(anchor.video_id)
        anchors_by_video[anchor.video_id].append(anchor.timestamp_ms)
        if sum(len(values) for values in anchors_by_video.values()) >= 24:
            break

    def load_video(video_id: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            rows = services.tv1.frames(video_id)
        except Exception:
            return video_id, []
        timestamps = anchors_by_video[video_id]
        scoped = [
            row for row in rows
            if isinstance(row, dict)
            and row.get("video_id") == video_id
            and row.get("preprocess_run_id") == services.preprocess_run_id
            and isinstance(row.get("frame_id"), int) and not isinstance(row.get("frame_id"), bool)
            and isinstance(row.get("timestamp_ms"), int) and not isinstance(row.get("timestamp_ms"), bool)
            and any(abs(int(row["timestamp_ms"]) - timestamp) <= window_ms for timestamp in timestamps)
        ] if isinstance(rows, list) else []
        return video_id, scoped

    by_video: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(
        max_workers=min(8, len(video_order)), thread_name_prefix="text-anchor-scope",
    ) as executor:
        for video_id, rows in executor.map(load_video, video_order):
            by_video[video_id] = rows
    scope: list[dict[str, object]] = []
    for video_id in video_order:
        for row in by_video.get(video_id, []):
            scope.append({"video_id": video_id, "frame_id": int(row["frame_id"])})
            if len(scope) >= max_frames:
                return scope
    return scope


def _search_visual_with_scope(
    services: "KisServices", query_id: str, query_text: str, candidate_frames: list[dict[str, object]],
) -> list[SearchCandidate]:
    # Global visual retrieval is always retained. Transcript scope is only a
    # bounded reranking hint and can never hide a correct silent/visual-only
    # result from the full corpus.
    global_candidates = services.visual.search(query_id, query_text)
    if not candidate_frames:
        return global_candidates
    try:
        scoped_candidates = services.visual.search(query_id, query_text, candidate_frames=candidate_frames)
    except TypeError:
        # Test doubles and legacy visual clients retain the previous public
        # signature. Global retrieval remains fully usable.
        return global_candidates
    except Exception:
        return global_candidates
    return _blend_global_and_scoped_visual(global_candidates, scoped_candidates)


def _search_visual_query_set_with_scope(
    services: "KisServices", query_id: str, query_texts: tuple[str, ...],
    candidate_frames: list[dict[str, object]], *, preserve_primary_order: bool = False,
    temporal_chain: bool = False, temporal_moment_start_index: int = 1,
) -> list[SearchCandidate]:
    """Search independent visible moments and reward cross-query agreement."""
    queries = tuple(dict.fromkeys(text.strip() for text in query_texts if text.strip()))
    if not queries:
        return []
    if len(queries) == 1:
        return _search_visual_with_scope(services, query_id, queries[0], candidate_frames)

    batch_search = getattr(services.visual, "search_batch", None)
    configured_limit = max(1, int(getattr(services.visual, "top_k", 100)))
    # WP03 already computes ``candidate_k_per_model`` FAISS hits.  Returning
    # that bounded pool to this in-process multi-moment fusion costs no extra
    # model encoding and prevents its own early top-100 cutoff from hiding a
    # video supported by different clauses at different timestamps.
    candidate_depth = max(
        configured_limit,
        int(getattr(services.visual, "candidate_k_per_model", configured_limit)),
    )
    if callable(batch_search):
        model_override = (
            tuple(getattr(services.visual, "temporal_models", ()))
            if temporal_chain
            else tuple(getattr(services.visual, "multi_moment_models", ()))
            if not preserve_primary_order
            else ()
        )
        try:
            batch_kwargs: dict[str, Any] = {"top_k": candidate_depth}
            if model_override:
                batch_kwargs["models"] = model_override
            if candidate_frames:
                batch_kwargs["candidate_frames"] = candidate_frames
            result_sets = batch_search(
                [
                    (f"{query_id}-visual-{index}", text, None)
                    for index, text in enumerate(queries)
                ],
                **batch_kwargs,
            )
        except TypeError:
            # Test doubles and older deployments expose the pre-depth API.
            result_sets = batch_search([
                (f"{query_id}-visual-{index}", text, None)
                for index, text in enumerate(queries)
            ])
        except Exception:
            result_sets = []
        if len(result_sets) == len(queries):
            return _blend_visual_query_results(
                result_sets,
                # Multi-moment KIS needs video-level agreement: the correct
                # video may contain each requested moment at a different
                # timestamp.  VQA's audited primary anchor stays conservative
                # unless its query plan explicitly opts into consensus.
                video_consensus_weight=0.25 if preserve_primary_order else 0.35,
                preserve_primary_order=preserve_primary_order,
                limit=candidate_depth,
                temporal_chain=temporal_chain,
                temporal_moment_start_index=temporal_moment_start_index,
            )

    def search_one(item: tuple[int, str]) -> list[SearchCandidate]:
        index, text = item
        try:
            # Scope only the first query. Applying it to every variant would
            # multiply GPU calls and let transcript evidence dominate all
            # visual moments of a long description.
            if index == 0:
                return _search_visual_with_scope(services, query_id, text, candidate_frames)
            try:
                return services.visual.search(query_id, text, top_k=candidate_depth)
            except TypeError:
                return services.visual.search(query_id, text)
        except Exception:
            return []

    with ThreadPoolExecutor(
        max_workers=min(3, len(queries)), thread_name_prefix="visual-query-set",
    ) as executor:
        results = list(executor.map(search_one, enumerate(queries)))
    return _blend_visual_query_results(
        results,
        video_consensus_weight=0.25 if preserve_primary_order else 0.35,
        preserve_primary_order=preserve_primary_order,
        limit=candidate_depth,
        temporal_chain=temporal_chain,
        temporal_moment_start_index=temporal_moment_start_index,
    )


def _temporal_chain_candidates(
    result_sets: list[list[SearchCandidate]], *, max_span_ms: int,
    moment_start_index: int = 1,
    max_candidates_per_video_per_moment: int = 32,
) -> list[SearchCandidate]:
    """Return the first frame of each complete, compact ordered moment chain.

    Query zero is the full-description anchor. Queries from
    ``moment_start_index`` onward are atomic moments in presentation order. A video must cover every atomic moment with
    strictly increasing timestamps; otherwise it receives no temporal bonus.
    """
    moment_sets = result_sets[moment_start_index:]
    if len(moment_sets) < 2 or max_span_ms <= 0:
        return []

    by_moment: list[dict[str, list[SearchCandidate]]] = []
    for candidates in moment_sets:
        by_video: dict[str, list[SearchCandidate]] = defaultdict(list)
        for candidate in candidates:
            if len(by_video[candidate.video_id]) < max_candidates_per_video_per_moment:
                by_video[candidate.video_id].append(candidate)
        by_moment.append(by_video)
    shared_videos = set(by_moment[0])
    for by_video in by_moment[1:]:
        shared_videos.intersection_update(by_video)
    if not shared_videos:
        return []

    anchor_rank_by_video: dict[str, int] = {}
    for candidate in result_sets[0]:
        anchor_rank_by_video.setdefault(candidate.video_id, candidate.rank)

    ranked_chains: list[tuple[float, int, int, SearchCandidate]] = []
    for video_id in shared_videos:
        states: list[tuple[float, int, tuple[SearchCandidate, ...]]] = [
            (1.0 / (60 + candidate.rank), candidate.timestamp_ms, (candidate,))
            for candidate in by_moment[0][video_id]
        ]
        for layer in by_moment[1:]:
            next_states: list[tuple[float, int, tuple[SearchCandidate, ...]]] = []
            for candidate in layer[video_id]:
                valid = [
                    state for state in states
                    if state[2][-1].timestamp_ms < candidate.timestamp_ms
                    and candidate.timestamp_ms - state[1] <= max_span_ms
                ]
                if not valid:
                    continue
                best = max(valid, key=lambda state: (state[0], -state[1]))
                next_states.append((
                    best[0] + 1.0 / (60 + candidate.rank),
                    best[1],
                    (*best[2], candidate),
                ))
            if not next_states:
                states = []
                break
            states = next_states
        if not states:
            continue
        chain_score, first_timestamp, chain = max(
            states,
            key=lambda state: (
                state[0],
                -(state[2][-1].timestamp_ms - state[1]),
                -state[1],
            ),
        )
        anchor_rank = anchor_rank_by_video.get(video_id, 1_000_000)
        if anchor_rank < 1_000_000:
            chain_score += 0.25 / (60 + anchor_rank)
        span_ms = chain[-1].timestamp_ms - first_timestamp
        first = chain[0]
        candidate = replace(
            first,
            score=chain_score,
            provenance={
                **first.provenance,
                "visual_temporal_chain_valid": True,
                "visual_temporal_chain": tuple({
                    "query_index": index + 1,
                    "rank": item.rank,
                    "frame_id": item.frame_id,
                    "timestamp_ms": item.timestamp_ms,
                } for index, item in enumerate(chain)),
                "visual_temporal_chain_span_ms": span_ms,
                "visual_temporal_chain_score": chain_score,
                "visual_temporal_anchor_rank": None if anchor_rank == 1_000_000 else anchor_rank,
            },
        )
        ranked_chains.append((chain_score, anchor_rank, span_ms, candidate))

    ranked_chains.sort(key=lambda item: (-item[0], item[1], item[2], item[3].video_id))
    return [
        replace(candidate, rank=rank)
        for rank, (_, _, _, candidate) in enumerate(ranked_chains, start=1)
    ]


def _temporal_cluster_candidates(
    result_sets: list[list[SearchCandidate]], *, max_span_ms: int,
    moment_start_index: int,
) -> list[SearchCandidate]:
    """Localize a compact partial sequence when one atomic query misses.

    Visual encoders are often reliable about scene co-occurrence but weak at
    camera-order language. This fallback requires the dedicated video locator
    (position one) plus at least two atomic moments in the same bounded window.
    It never claims a strict chain and cannot activate for the legacy query
    layout where no locator role is present.
    """
    if moment_start_index < 2 or len(result_sets) <= moment_start_index or max_span_ms <= 0:
        return []
    locator_index = moment_start_index - 1
    atomic_indexes = tuple(range(moment_start_index, len(result_sets)))
    required_atomic_support = min(len(atomic_indexes), max(2, (2 * len(atomic_indexes) + 2) // 3))

    by_query_video: list[dict[str, list[SearchCandidate]]] = []
    for candidates in result_sets:
        grouped: dict[str, list[SearchCandidate]] = defaultdict(list)
        for candidate in candidates:
            if len(grouped[candidate.video_id]) < 32:
                grouped[candidate.video_id].append(candidate)
        by_query_video.append(grouped)

    anchor_rank_by_video = {
        candidate.video_id: candidate.rank for candidate in result_sets[0]
    }
    candidates_by_video: list[tuple[float, int, int, SearchCandidate]] = []
    for video_id, locator_candidates in by_query_video[locator_index].items():
        timeline = sorted({
            candidate.timestamp_ms
            for query_index in (locator_index, *atomic_indexes)
            for candidate in by_query_video[query_index].get(video_id, ())
        })
        best_window: tuple[float, int, int, dict[int, SearchCandidate]] | None = None
        for start_ms in timeline:
            selected: dict[int, SearchCandidate] = {}
            for query_index in (locator_index, *atomic_indexes):
                in_window = [
                    candidate for candidate in by_query_video[query_index].get(video_id, ())
                    if start_ms <= candidate.timestamp_ms <= start_ms + max_span_ms
                ]
                if in_window:
                    selected[query_index] = min(
                        in_window, key=lambda candidate: (candidate.rank, candidate.timestamp_ms),
                    )
            if locator_index not in selected:
                continue
            atomic_support = sum(index in selected for index in atomic_indexes)
            if atomic_support < required_atomic_support:
                continue
            selected_values = tuple(selected.values())
            span_ms = max(item.timestamp_ms for item in selected_values) - min(
                item.timestamp_ms for item in selected_values
            )
            score = sum(1.0 / (60 + item.rank) for item in selected_values)
            anchor_rank = anchor_rank_by_video.get(video_id)
            if anchor_rank is not None:
                score += 0.10 / (60 + anchor_rank)
            proposal = (score, atomic_support, span_ms, selected)
            if best_window is None or (score, atomic_support, -span_ms) > (
                best_window[0], best_window[1], -best_window[2]
            ):
                best_window = proposal
        if best_window is None:
            continue
        score, atomic_support, span_ms, selected = best_window
        representative = min(
            selected.values(), key=lambda candidate: (candidate.timestamp_ms, candidate.rank),
        )
        localized = replace(
            representative,
            score=score,
            provenance={
                **representative.provenance,
                "visual_temporal_cluster_valid": True,
                "visual_temporal_cluster": tuple({
                    "query_index": query_index,
                    "rank": selected[query_index].rank,
                    "frame_id": selected[query_index].frame_id,
                    "timestamp_ms": selected[query_index].timestamp_ms,
                } for query_index in sorted(selected)),
                "visual_temporal_cluster_span_ms": span_ms,
                "visual_temporal_cluster_moment_support": atomic_support,
                "visual_temporal_cluster_moment_count": len(atomic_indexes),
                "visual_temporal_cluster_score": score,
            },
        )
        candidates_by_video.append((score, atomic_support, span_ms, localized))

    candidates_by_video.sort(
        key=lambda item: (-item[0], -item[1], item[2], item[3].video_id),
    )
    return [
        replace(candidate, rank=rank)
        for rank, (_, _, _, candidate) in enumerate(candidates_by_video, start=1)
    ]


def _blend_visual_query_results(
    result_sets: list[list[SearchCandidate]], *, bucket_ms: int = 5_000,
    video_consensus_weight: float = 0.25, preserve_primary_order: bool = False,
    limit: int = 100, primary_reserve: int = 20,
    temporal_chain: bool = False, temporal_max_span_ms: int = 60_000,
    temporal_moment_start_index: int = 1,
) -> list[SearchCandidate]:
    """Fuse visible moments while retaining independently strong frames.

    Exact temporal agreement receives all query votes. Different moments in
    the same video receive a smaller video-level consensus bonus, which is
    useful for long KIS descriptions without turning video identity into a
    hard gate.
    """
    if len(result_sets) <= 1:
        return result_sets[0] if result_sets else []
    per_video_query: dict[str, dict[int, float]] = defaultdict(dict)
    per_video_candidates: dict[str, dict[int, list[SearchCandidate]]] = defaultdict(lambda: defaultdict(list))
    buckets: dict[tuple[str, int], dict[str, Any]] = {}
    for query_index, candidates in enumerate(result_sets):
        for candidate in candidates:
            contribution = 1.0 / (60 + candidate.rank)
            per_video_candidates[candidate.video_id][query_index].append(candidate)
            previous = per_video_query[candidate.video_id].get(query_index, 0.0)
            per_video_query[candidate.video_id][query_index] = max(previous, contribution)
            key = (candidate.video_id, candidate.timestamp_ms // bucket_ms)
            entry = buckets.setdefault(
                key,
                {
                    "candidate": candidate, "query_contributions": {},
                    "query_indices": set(), "query_ranks": {},
                },
            )
            entry["query_ranks"][query_index] = min(
                candidate.rank, entry["query_ranks"].get(query_index, candidate.rank),
            )
            current = entry["query_contributions"].get(query_index, 0.0)
            if contribution > current:
                entry["query_contributions"][query_index] = contribution
                entry["query_indices"].add(query_index)
            if contribution > 1.0 / (60 + entry["candidate"].rank):
                entry["candidate"] = candidate

    ordered: list[tuple[float, dict[str, Any]]] = []
    for entry in buckets.values():
        candidate = entry["candidate"]
        exact_score = sum(entry["query_contributions"].values())
        video_score = sum(per_video_query[candidate.video_id].values())
        ordered_score = _ordered_supplemental_support(
            per_video_candidates[candidate.video_id], len(result_sets),
        )
        score = (
            exact_score
            + video_consensus_weight * max(0.0, video_score - exact_score)
            + video_consensus_weight * ordered_score
        )
        entry["ordered_support_score"] = ordered_score
        ordered.append((score, entry))
    if preserve_primary_order:
        # For VQA the complete description is the audited video-identifying
        # anchor. Moment queries improve recall but a generic supplemental
        # phrase (for example "autonomous car") must not reorder a rank-1
        # exact anchor behind unrelated videos. Supplemental-only hits remain
        # available after every primary hit.
        ordered.sort(
            key=lambda item: (
                0 if 0 in item[1]["query_ranks"] else 1,
                item[1]["query_ranks"].get(0, 1_000_000),
                -item[0], item[1]["candidate"].video_id,
                item[1]["candidate"].timestamp_ms, item[1]["candidate"].frame_id,
            )
        )
    else:
        ordered.sort(
            key=lambda item: (
                -item[0], item[1]["candidate"].video_id,
                item[1]["candidate"].timestamp_ms, item[1]["candidate"].frame_id,
            )
        )
        # Preserve a bounded quota from the complete-description anchor, then
        # let strong moment/co-occurrence evidence compete for the remaining
        # slots. Reserving the entire primary pool accidentally discarded all
        # decomposed-query hits whenever WP03 returned a full candidate list.
        primary = [item for item in ordered if 0 in item[1]["query_ranks"]]
        reserved_primary = primary[: min(len(primary), max(0, primary_reserve), limit)]
        reserved_keys = {
            (item[1]["candidate"].video_id, item[1]["candidate"].timestamp_ms // bucket_ms)
            for item in reserved_primary
        }
        ordered = reserved_primary + [
            item for item in ordered
            if (
                item[1]["candidate"].video_id,
                item[1]["candidate"].timestamp_ms // bucket_ms,
            ) not in reserved_keys
        ][: max(0, limit - len(reserved_primary))]
        # This list feeds another multimodal fusion stage whose RRF uses the
        # visual rank. Promote each video's strongest visual representative
        # before duplicate time buckets, otherwise repeated shots from a few
        # easy videos can make a rank-17 hit in a distinctive query look like
        # a rank-300 visual vote downstream.
        representatives: list[tuple[float, dict[str, Any]]] = []
        duplicates: list[tuple[float, dict[str, Any]]] = []
        video_order: list[str] = []
        by_video: dict[str, list[tuple[int, tuple[float, dict[str, Any]]]]] = defaultdict(list)
        for position, item in enumerate(ordered):
            video_id = item[1]["candidate"].video_id
            if video_id not in by_video:
                video_order.append(video_id)
            by_video[video_id].append((position, item))
        for video_id in video_order:
            group = by_video[video_id]
            _, representative = min(
                group,
                key=lambda positioned: (
                    0 if {0, 1} <= positioned[1][1]["query_indices"]
                    else 1 if 1 in positioned[1][1]["query_indices"]
                    else 2,
                    positioned[0],
                ),
            )
            representatives.append(representative)
            duplicates.extend(
                item for _, item in group if item is not representative
            )
        ordered = representatives + duplicates
    blended = [
        replace(
            entry["candidate"], rank=rank, score=score,
            provenance={
                **entry["candidate"].provenance,
                "visual_query_variants": tuple(sorted(entry["query_indices"])),
                "visual_video_support_count": len(per_video_query[entry["candidate"].video_id]),
                "visual_ordered_support_score": entry.get("ordered_support_score", 0.0),
            },
        )
        for rank, (score, entry) in enumerate(ordered[:limit], start=1)
    ]
    if not temporal_chain:
        return blended
    chains = _temporal_chain_candidates(
        result_sets, max_span_ms=temporal_max_span_ms,
        moment_start_index=temporal_moment_start_index,
    )
    clusters = _temporal_cluster_candidates(
        result_sets, max_span_ms=min(30_000, temporal_max_span_ms),
        moment_start_index=temporal_moment_start_index,
    )
    if not chains and not clusters:
        return blended
    # A complete chain found only in the deep tail is weaker evidence than a
    # tight locator/scene cluster whose clauses all rank near the top.  A
    # binary "all chains before all clusters" rule made accidental chains
    # from a 1,000-candidate recall pool bury the correct q15 scene.  Compare
    # their reciprocal-rank confidence instead and retain the strongest
    # localization per video.
    localized_by_video: dict[str, SearchCandidate] = {}
    for candidate in (*chains, *clusters):
        previous = localized_by_video.get(candidate.video_id)
        if previous is None or (
            candidate.score,
            bool(candidate.provenance.get("visual_temporal_chain_valid")),
            -candidate.timestamp_ms,
        ) > (
            previous.score,
            bool(previous.provenance.get("visual_temporal_chain_valid")),
            -previous.timestamp_ms,
        ):
            localized_by_video[candidate.video_id] = candidate
    localized = sorted(
        localized_by_video.values(),
        key=lambda candidate: (
            -candidate.score,
            -int(bool(candidate.provenance.get("visual_temporal_chain_valid"))),
            candidate.timestamp_ms,
            candidate.video_id,
        ),
    )
    localized_videos = {candidate.video_id for candidate in localized}
    combined = localized + [
        candidate for candidate in blended if candidate.video_id not in localized_videos
    ]
    return [
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(combined[:limit], start=1)
    ]


def _ordered_supplemental_support(
    by_query: dict[int, list[SearchCandidate]], query_count: int,
) -> float:
    """Best non-decreasing temporal chain across supplemental moments."""
    supplemental = [index for index in range(1, query_count) if by_query.get(index)]
    if len(supplemental) < 2:
        return 0.0
    states: list[tuple[int, float]] = [
        (candidate.timestamp_ms, 1.0 / (60 + candidate.rank))
        for candidate in by_query[supplemental[0]]
    ]
    covered = 1
    for query_index in supplemental[1:]:
        next_states: list[tuple[int, float]] = []
        for candidate in by_query[query_index]:
            previous_scores = [score for timestamp, score in states if timestamp <= candidate.timestamp_ms]
            if previous_scores:
                next_states.append((
                    candidate.timestamp_ms,
                    max(previous_scores) + 1.0 / (60 + candidate.rank),
                ))
        if not next_states:
            return 0.0
        states = next_states
        covered += 1
    return max((score for _, score in states), default=0.0) if covered == len(supplemental) else 0.0


def _blend_global_and_scoped_visual(
    global_candidates: list[SearchCandidate], scoped_candidates: list[SearchCandidate],
    *, scoped_weight: float = 0.5, bucket_ms: int = 1_000,
) -> list[SearchCandidate]:
    """Rerank global visual hits with bounded ASR support without hard gating."""
    if not scoped_candidates:
        return global_candidates
    buckets: dict[tuple[str, int], dict[str, Any]] = {}
    for origin, weight, candidates in (
        ("global", 1.0, global_candidates),
        ("transcript_scope", scoped_weight, scoped_candidates),
    ):
        for candidate in candidates:
            key = (candidate.video_id, candidate.timestamp_ms // bucket_ms)
            entry = buckets.setdefault(key, {"candidate": candidate, "score": 0.0, "origins": set()})
            entry["score"] += weight / (60 + candidate.rank)
            entry["origins"].add(origin)
            # Prefer the globally retrieved representative when both searches
            # address the same time bucket; it is independent of ASR.
            if origin == "global" and "global" not in entry.get("selected_from", set()):
                entry["candidate"] = candidate
                entry["selected_from"] = {"global"}
    ordered = sorted(
        buckets.values(),
        key=lambda entry: (
            -float(entry["score"]),
            entry["candidate"].video_id,
            entry["candidate"].timestamp_ms,
            entry["candidate"].frame_id,
        ),
    )
    return [
        replace(
            entry["candidate"], rank=rank, score=float(entry["score"]),
            provenance={
                **entry["candidate"].provenance,
                "visual_search_origins": tuple(sorted(entry["origins"])),
                "transcript_scope_supported": "transcript_scope" in entry["origins"],
            },
        )
        for rank, entry in enumerate(ordered, start=1)
    ]


def _canonicalize_branch_results(
    branch_results: dict[str, list[SearchCandidate]], services: "KisServices",
) -> dict[str, list[SearchCandidate]]:
    """Map every retrieval identity onto the selected frames owned by TV1.

    Visual, OCR, object and metadata artifacts may carry an original-frame ID
    that was not selected as a keyframe in the active run (ASR can do the same
    on older producers). Returning that ID makes TV5 request a thumbnail which
    correctly does not exist. Resolve all branches, not just visual, and load
    TV1 authority once per video so the fix remains bounded for top-100 search.
    """
    if not branch_results:
        return {}

    video_ids = tuple(dict.fromkeys(
        candidate.video_id
        for candidates in branch_results.values()
        for candidate in candidates
    ))
    if not video_ids:
        return {branch: [] for branch in branch_results}

    def load_frames(video_id: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            payload = services.tv1.frames(video_id)
        except Exception:
            return video_id, []
        frames = [
            frame for frame in payload
            if isinstance(frame, dict)
            and frame.get("video_id") == video_id
            and frame.get("preprocess_run_id") == services.preprocess_run_id
            and isinstance(frame.get("frame_id"), int) and not isinstance(frame.get("frame_id"), bool)
            and isinstance(frame.get("timestamp_ms"), int) and not isinstance(frame.get("timestamp_ms"), bool)
        ] if isinstance(payload, list) else []
        return video_id, frames

    frames_by_video: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(video_ids)), thread_name_prefix="tv1-frame-authority") as executor:
        for video_id, frames in executor.map(load_frames, video_ids):
            frames_by_video[video_id] = frames

    canonical: dict[str, list[SearchCandidate]] = {}
    for branch, candidates in branch_results.items():
        mapped: list[SearchCandidate] = []
        for candidate in candidates:
            frames = frames_by_video.get(candidate.video_id, [])
            if not frames:
                continue
            frame = next((item for item in frames if item["frame_id"] == candidate.frame_id), None)
            mapping = "exact_frame_id"
            if frame is None:
                frame = min(
                    frames,
                    key=lambda item: (
                        abs(int(item["timestamp_ms"]) - candidate.timestamp_ms),
                        int(item["timestamp_ms"]),
                        int(item["frame_id"]),
                    ),
                )
                mapping = "nearest_timestamp"

            keyframe_path = frame.get("keyframe_path")
            evidence_refs = tuple(
                ref for ref in candidate.evidence_refs
                if not (isinstance(ref, str) and ref.startswith("keyframe:"))
            )
            if isinstance(keyframe_path, str) and keyframe_path:
                evidence_refs = (*evidence_refs, f"keyframe:{keyframe_path}")
            source_provenance = {
                "source_frame_id": candidate.frame_id,
                "source_timestamp_ms": candidate.timestamp_ms,
            }
            if branch == "visual":
                # Retain the established visual-specific audit fields.
                source_provenance.update({
                    "visual_source_frame_id": candidate.frame_id,
                    "visual_source_timestamp_ms": candidate.timestamp_ms,
                })
            mapped.append(replace(
                candidate,
                frame_id=int(frame["frame_id"]),
                timestamp_ms=int(frame["timestamp_ms"]),
                evidence_refs=evidence_refs,
                provenance={
                    **candidate.provenance,
                    **source_provenance,
                    "frame_identity_mapping": mapping,
                    "frame_identity_delta_ms": abs(int(frame["timestamp_ms"]) - candidate.timestamp_ms),
                },
                preprocess_run_id=services.preprocess_run_id,
            ))
        canonical[branch] = mapped
    return canonical


def _canonicalize_visual_candidates(
    candidates: list[SearchCandidate], services: "KisServices",
) -> list[SearchCandidate]:
    """Backward-compatible wrapper for callers/tests of the old helper."""
    return _canonicalize_branch_results({"visual": candidates}, services)["visual"]


def _external_vlm_rerank(
    ranked: list[SearchCandidate], query_text: str, services: "KisServices",
) -> list[SearchCandidate]:
    """Apply a bounded VLM bonus without letting it replace retrieval.

    An external verifier can be confidently wrong. Its score therefore adds
    at most one percent of the best fusion score and never removes a candidate.
    """
    verifier = ExternalVlmVerifier.from_environment()
    root = services.derivative_asset_root
    if not verifier.enabled or root is None or not ranked:
        return ranked
    root = root.resolve()
    inspectable: list[tuple[int, Path]] = []
    for candidate_index, candidate in enumerate(ranked):
        reference = next((value for value in candidate.evidence_refs if value.startswith("keyframe:")), None)
        if reference is None:
            continue
        path = (root / reference.removeprefix("keyframe:")).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file() and path.suffix.lower() in services.allowed_image_extensions:
            inspectable.append((candidate_index, path))
        if len(inspectable) == verifier.max_images:
            break
    verdict = verifier.rank(query_text, [path for _, path in inspectable])
    if not verdict:
        return ranked
    scores = {inspectable[local_index][0]: score for local_index, score in verdict}
    best_fusion_score = max((candidate.score or 0.0 for candidate in ranked), default=0.0)
    max_bonus = best_fusion_score * 0.01
    scored: list[tuple[float, int, SearchCandidate]] = []
    for index, candidate in enumerate(ranked):
        vlm_score = scores.get(index)
        bonus = max_bonus * (vlm_score / 100.0) if vlm_score is not None else 0.0
        scored.append((
            (candidate.score or 0.0) + bonus,
            index,
            replace(
                candidate,
                provenance={
                    **candidate.provenance,
                    **({
                        "external_vlm_score": vlm_score,
                        "external_vlm_rrf_bonus": bonus,
                    } if vlm_score is not None else {}),
                },
            ),
        ))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        replace(candidate, rank=rank)
        for rank, (_, _, candidate) in enumerate(scored, start=1)
    ]


def _vqa_ranked_candidates(
    branch_results: dict[str, list[SearchCandidate]], *, question: str | None,
    description: str = "", weights: dict[str, float] | None = None, top_k: int,
) -> list[SearchCandidate]:
    """Fuse normally, but put actual ASR hits first for textual VQA questions.

    Visual/OCR hits are still retained as fallback. They must not outrank a
    segment that directly names the answer to a question such as "ở đâu?".
    """
    # The VQA description identifies the video visually; OCR/ASR/object text
    # is primarily answer evidence after that video has been found. Equal RRF
    # weights let several generic transcript/object hits bury a rank-1 visual
    # match. Spoken-answer questions are handled by the explicit ASR-first
    # path below, so other VQA queries receive a bounded visual prior.
    effective_weights = dict(weights or {})
    plan = build_qa_query_plan(description, question or "")
    if plan.operation == "read_chart_relation":
        # The chart itself answers the question, but OCR/lecture narration are
        # strong video-identification evidence.  Keep all three modalities
        # active instead of suppressing text merely because the final value is
        # visual.
        effective_weights["visual"] = 1.5 * float(effective_weights.get("visual", 1.0))
        effective_weights["ocr"] = 1.5 * float(effective_weights.get("ocr", 1.0))
        effective_weights["asr"] = float(effective_weights.get("asr", 1.0))
    elif not _question_prefers_asr(question, description):
        effective_weights["visual"] = 2.0 * float(effective_weights.get("visual", 1.0))
        effective_weights["asr"] = 0.25 * float(effective_weights.get("asr", 1.0))
    fused = fuse_kis(branch_results, weights=effective_weights, top_k=top_k)
    asr = branch_results.get("asr", [])
    if not asr or not _question_prefers_asr(question, description):
        return fused
    asr_ranked = fuse_kis({"asr": asr}, weights=effective_weights, top_k=top_k)
    seen = {(candidate.video_id, candidate.timestamp_ms // 1_000) for candidate in asr_ranked}
    ordered = asr_ranked + [
        candidate for candidate in fused
        if (candidate.video_id, candidate.timestamp_ms // 1_000) not in seen
    ]
    return [replace(candidate, rank=index) for index, candidate in enumerate(ordered[:top_k], start=1)]


def _rank_multimoment_kis(
    branch_results: dict[str, list[SearchCandidate]],
    *,
    weights: dict[str, float] | None,
    top_k: int,
) -> list[SearchCandidate]:
    """Rerank decomposed KIS while preserving visual recall and video diversity.

    Multi-scene descriptions are retrieved from several visual clauses.  OCR,
    ASR and metadata are useful corroboration at *video* level, but their
    frame timestamps do not necessarily coincide with the visual shot.  A
    normal frame-level top-100 can therefore be filled by repeated shots from
    a few videos and evict a correct result supplied by another branch.  This
    policy fuses the complete bounded branch pool, then presents one
    representative per video before duplicate shots.
    """
    pool_size = max(top_k, sum(len(candidates) for candidates in branch_results.values()))
    deep_fused = fuse_kis(branch_results, weights=weights, top_k=pool_size)

    # Multi-scene KIS is identified by visible co-occurrence/order. OCR/ASR can
    # corroborate a visually retrieved video, but generic lecture transcripts
    # must not replace the whole visual Recall@K set. Anchor the first K unique
    # visual videos, attach their best multimodal score, then use text-only
    # videos solely to fill any remaining capacity.
    visual_representatives: list[SearchCandidate] = []
    visual_videos: set[str] = set()
    for candidate in branch_results.get("visual", []):
        if candidate.video_id in visual_videos:
            continue
        visual_videos.add(candidate.video_id)
        visual_representatives.append(candidate)
        if len(visual_representatives) >= top_k:
            break

    best_fused_by_video: dict[str, SearchCandidate] = {}
    for candidate in deep_fused:
        best_fused_by_video.setdefault(candidate.video_id, candidate)

    representatives: list[SearchCandidate] = []
    for candidate in visual_representatives:
        fused = best_fused_by_video.get(candidate.video_id)
        representatives.append(replace(
            candidate,
            source="fusion" if fused is not None else candidate.source,
            score=fused.score if fused is not None else candidate.score,
            model_scores=(
                {**candidate.model_scores, **fused.model_scores}
                if fused is not None else candidate.model_scores
            ),
            provenance_sources=(
                fused.provenance_sources if fused is not None
                else candidate.provenance_sources
            ),
            provenance={
                **candidate.provenance,
                "multimoment_video_diversified": True,
                "multimoment_pre_diversity_rank": (
                    fused.rank if fused is not None else candidate.rank
                ),
                "multimoment_visual_anchor_rank": candidate.rank,
            },
        ))

    seen_videos = {candidate.video_id for candidate in representatives}
    for candidate in deep_fused:
        if candidate.video_id in seen_videos:
            continue
        seen_videos.add(candidate.video_id)
        representatives.append(replace(
            candidate,
            provenance={
                **candidate.provenance,
                "multimoment_video_diversified": True,
                "multimoment_pre_diversity_rank": candidate.rank,
            },
        ))
        if len(representatives) >= top_k:
            break
    # Preserve other moments from the same video. Reserve at most one quarter
    # of the output for those windows without letting repeated nearby frames
    # displace the diverse video shortlist.
    seen_windows = {(c.video_id, c.timestamp_ms // 1_000) for c in representatives}
    duplicates: list[SearchCandidate] = []
    for candidate in deep_fused:
        identity = (candidate.video_id, candidate.timestamp_ms // 1_000)
        if identity in seen_windows:
            continue
        seen_windows.add(identity)
        duplicates.append(candidate)
    reserve = min(len(duplicates), max(0, top_k // 4))
    representative_budget = max(1, top_k - reserve)
    ordered = representatives[:representative_budget] + duplicates[:reserve]
    ordered += representatives[representative_budget:] + duplicates[reserve:]
    return [
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(ordered[:top_k], start=1)
    ]


@dataclass
class KisServices:
    tv1: TV1Client
    tv3: TV3Client
    visual: TV2VisualClient
    refine: TV2RefineClient | None = None
    feedback: Any | None = None
    dense_ocr: Any | None = None
    preprocess_run_id: str = "unknown"
    original_media_root: Path | None = None
    media_registry: Mapping[str, object] | None = None
    allowed_media_extensions: frozenset[str] = frozenset()
    derivative_asset_root: Path | None = None
    allowed_image_extensions: frozenset[str] = frozenset()


def run_kis_query(
    query_text: str,
    services: KisServices,
    *,
    query_id: str | None = None,
    top_k: int = 100,
    weights: dict[str, float] | None = None,
    visual_query_text: str | None = None,
    visual_query_texts: tuple[str, ...] = (),
    task: Literal["KIS", "VQA"] = "KIS",
    question: str | None = None,
    additional_branches: tuple[str, ...] = (),
    preserve_visual_primary_order: bool | None = None,
) -> list[SearchCandidate]:
    """Retrieve from every branch while allowing a visual-only query expansion.

    The TV3 textual indexes are queried with the original user language. A
    separate English expansion can be passed for CLIP-style visual retrieval
    without degrading OCR/ASR matching or changing the external API contract.
    """
    if task == "VQA":
        if not question or not question.strip():
            raise ValueError("VQA retrieval requires a non-empty question")
        # The event description identifies the video; the question is used by
        # the grounded answerer after retrieval. Feeding question words such
        # as "ở đâu" into the lexical index makes generic language outweigh
        # the actual event and promotes unrelated OCR/ASR rows.
        retrieval_description = _description_for_vqa_retrieval(query_text, question)
        decision = route_qa(retrieval_description, question, query_id=query_id, limit=top_k)
    else:
        decision = route_kis(query_text, query_id=query_id, limit=top_k)
    if additional_branches:
        allowed = {"visual", "ocr", "asr", "object", "metadata"}
        decision = replace(
            decision,
            branches=tuple(dict.fromkeys((*decision.branches, *(branch for branch in additional_branches if branch in allowed)))),
        )
    visual_queries = tuple(text.strip() for text in visual_query_texts if text.strip())
    if not visual_queries:
        visual_queries = (
            visual_query_text.strip() if visual_query_text and visual_query_text.strip() else query_text,
        )
    preserve_primary = (
        task == "VQA"
        if preserve_visual_primary_order is None
        else bool(preserve_visual_primary_order)
    )
    temporal_query = task == "KIS" and is_temporal_visual_query(query_text)
    multi_moment_query = task == "KIS" and len(split_visual_moments(query_text)) >= 2

    branch_results: dict[str, list[SearchCandidate]] = {}
    tv3_routes = [b for b in decision.branches if b in {"ocr", "asr", "object", "metadata"}]
    transcript_scope, transcript_candidates = (
        _transcript_frame_scope(services, query_id=decision.request.query_id, query_text=retrieval_description if task == "VQA" else query_text)
        if task == "VQA" and "asr" in decision.branches
        else ([], [])
    )
    # Symbolic illustrations in lecture slides often name the surrounding
    # spoken topic more reliably than they match a corpus-wide image vector.
    # For example, tug-of-war is a conventional symbol for competition.  Use
    # a small audited concept expansion to find transcript anchors first, then
    # let the visual model choose the exact nearby keyframe.  The global ANN
    # results are still retained by WP03, so a bad/missing transcript cannot
    # hide a genuinely visual-only result.
    if task == "KIS" and multi_moment_query and "asr" in tv3_routes:
        locator_query = semantic_text_locator_query(query_text)
        if locator_query:
            locator_request = replace(
                decision.request, query_text=locator_query, limit=min(top_k, 100),
            )
            try:
                locator_candidates = services.tv3.search("asr", locator_request)
            except Exception:
                locator_candidates = []
            if locator_candidates:
                branch_results["asr"] = locator_candidates
                tv3_routes = [route for route in tv3_routes if route != "asr"]
                transcript_scope = _frame_scope_around_text_anchors(
                    services, locator_candidates,
                )
    if transcript_candidates:
        branch_results["asr"] = transcript_candidates
        tv3_routes = [route for route in tv3_routes if route != "asr"]
    # WP03 runs in a separate process and TV3 exposes independent indexes.
    # They do not share inputs, so serializing them turns request latency into
    # the sum of both backends. Joining in this fixed order keeps fusion
    # deterministic.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="kis-retrieval") as executor:
        visual_future = (
            executor.submit(
                _search_visual_query_set_with_scope, services, decision.request.query_id,
                visual_queries, transcript_scope, preserve_primary_order=preserve_primary,
                temporal_chain=temporal_query,
                temporal_moment_start_index=2 if temporal_query else 1,
            )
            if "visual" in decision.branches else None
        )
        tv3_future = (
            executor.submit(services.tv3.search_all, decision.request, tv3_routes)
            if tv3_routes else None
        )
        if visual_future is not None:
            try:
                branch_results["visual"] = visual_future.result()
            except Exception:
                # A visual-worker failure must not discard usable textual
                # evidence from TV3.
                branch_results["visual"] = []
        if tv3_future is not None:
            branch_results.update(tv3_future.result())

    if branch_results:
        branch_results = _canonicalize_branch_results(branch_results, services)

    ranked = (
        _vqa_ranked_candidates(
            branch_results, question=question, description=retrieval_description,
            weights=weights, top_k=top_k,
        )
        if task == "VQA"
        else (
            _rank_multimoment_kis(branch_results, weights=weights, top_k=top_k)
            if multi_moment_query
            else fuse_kis(branch_results, weights=weights, top_k=top_k)
        )
    )
    ranked = [replace(c, query_id=decision.request.query_id, preprocess_run_id=services.preprocess_run_id) for c in ranked]
    ranked = _external_vlm_rerank(ranked, query_text, services)

    # Exact decoding is deliberately reserved for the leading KIS result.
    # Previously every KIS and VQA request launched up to five independent
    # WP09 decoders, which dominated latency and gave VQA no extra evidence.
    if task == "KIS" and services.refine is not None:
        ranked = _refine_top(ranked, query_text, services, refine_top_n=1)
    return ranked


def _refine_top(ranked: list[SearchCandidate], query_text: str, services: KisServices, refine_top_n: int = 5) -> list[SearchCandidate]:
    """Ask WP09 to tighten the frame for the top few candidates only.

    Refinement is comparatively expensive (it decodes the original video),
    so it is only spent on the handful of candidates most likely to become
    the actual submission, matching the plan's "shared exact-frame service
    improves the real frame for KIS/VQA/TRAKE" description.
    """
    out = list(ranked)
    if services.original_media_root is None:
        return out
    from .media_identity import resolve_original_media_path
    for i, c in enumerate(out[:refine_top_n]):
        try:
            media_path = resolve_original_media_path(services.original_media_root, c.video_id, services.media_registry, services.preprocess_run_id, services.allowed_media_extensions)
        except ValueError:
            continue
        result = services.refine.refine(
            {
                "candidate": {"video_id": c.video_id, "frame_id": c.frame_id, "timestamp_ms": c.timestamp_ms,
                              "upstream_score": c.score, "confidence": c.confidence},
                "video_path": str(media_path),
                "task": "KIS",
                "refinement_text": query_text,
                "policy": "representative",
                "context": {
                    "preprocess_run_id": services.preprocess_run_id,
                    "media_record_ref": c.video_id,
                    "mapping_ref": f"tv1-frames/{c.video_id}",
                    "decoder_version": "pyav-tv4-adapter-1",
                    "model_version": "n/a",
                    "config_version": "default",
                },
                "decode_budget": {"max_decoded_frames": 64, "max_window_ms": 4000, "max_decode_time_ms": 8000, "max_dense_regions": 3},
            }
        )
        if not result or not result.get("hypotheses"):
            continue
        best = canonical_refined_candidate(result["hypotheses"][0])
        if best is not None:
            out[i] = replace(c, frame_id=best[0], timestamp_ms=best[1])
    return out


def canonical_refined_candidate(payload: object) -> tuple[int, int] | None:
    """Allow a replacement only when WP09 explicitly proved original identity."""
    if not isinstance(payload, dict):
        return None
    if payload.get("mapping_guaranteed") is not True or payload.get("submission_selectable") is not True:
        return None
    if payload.get("provenance_mode") != "live" or payload.get("identity_source") != "certified_run_consecutive_original_decode":
        return None
    if payload.get("media_identity_verified") is not True or payload.get("producer_compatibility_verified") is not True:
        return None
    if not isinstance(payload.get("certification_id"), str) or not isinstance(payload.get("certification_report_sha256"), str):
        return None
    frame_id, timestamp_ms = payload.get("frame_id"), payload.get("timestamp_ms")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        return None
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
        return None
    return frame_id, timestamp_ms
