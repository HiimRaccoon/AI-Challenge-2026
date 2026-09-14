"""WP12 — TRAKE: retrieve the video, then align an ordered event sequence to
non-decreasing-in-time semantic keyframes inside it.

Two alignment strategies are implemented, per the plan's "benchmark greedy
vs chain vs DP vs beam" requirement:

  * `align_greedy`   — pick each event's best candidate independently, then
                        repair any timestamp inversions greedily. Fast,
                        used as the always-available baseline.
  * `align_dp`        — exact DP: maximize sum(event scores) subject to
                        frame_1.ts < frame_2.ts < ... < frame_n.ts inside
                        the chosen video. This is the "chain/DP" strategy
                        and is what TV4 submits unless it degrades (an event
                        has zero candidates).

Both operate on per-event candidate pools TV4 already fused with WP10 for
that single retrieved video, so they never need TV2/TV3 to run again.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import SearchCandidate, TrakeHypothesis


@dataclass(frozen=True)
class EventPool:
    event_index: int
    candidates: tuple[SearchCandidate, ...]  # already filtered to one video, sorted by score desc


def pick_video(stage1_ranked: list[SearchCandidate]) -> str | None:
    """Stage 1 (Retrieval): the video of the top-ranked fused candidate."""
    return stage1_ranked[0].video_id if stage1_ranked else None


def build_event_pools(video_id: str, per_event_candidates: list[list[SearchCandidate]]) -> list[EventPool]:
    pools = []
    for idx, cands in enumerate(per_event_candidates):
        same_video = tuple(sorted((c for c in cands if c.video_id == video_id), key=lambda c: -(c.score or 0.0)))
        pools.append(EventPool(event_index=idx, candidates=same_video))
    return pools


def align_greedy(pools: list[EventPool]) -> list[SearchCandidate] | None:
    if any(not p.candidates for p in pools):
        return None
    chosen = [p.candidates[0] for p in pools]
    # Repair ordering left-to-right: each later event needs a later frame.
    for i in range(1, len(chosen)):
        if chosen[i].timestamp_ms <= chosen[i - 1].timestamp_ms:
            better = next((c for c in pools[i].candidates if c.timestamp_ms > chosen[i - 1].timestamp_ms), None)
            if better is not None:
                chosen[i] = better
            else:
                return None
    return chosen


def align_dp(pools: list[EventPool], max_candidates_per_event: int = 20) -> list[SearchCandidate] | None:
    """Exact DP maximizing total score subject to strictly increasing timestamps.

    O(sum |pool_i| * |pool_{i-1}|); pools are pre-truncated to the top
    `max_candidates_per_event` (already score-sorted) to keep this bounded
    for pathological events with hundreds of candidates.
    """
    if any(not p.candidates for p in pools):
        return None
    trimmed = [p.candidates[:max_candidates_per_event] for p in pools]

    # dp[i][j] = (best total score using trimmed[i][j] as event i's frame, backpointer)
    dp: list[list[tuple[float, int]]] = [
        [(c.score or 0.0, -1) for c in trimmed[0]]
    ]
    for i in range(1, len(trimmed)):
        row: list[tuple[float, int]] = []
        for j, c in enumerate(trimmed[i]):
            best_score, best_prev = float("-inf"), -1
            for k, prev_c in enumerate(trimmed[i - 1]):
                if prev_c.timestamp_ms >= c.timestamp_ms:
                    continue
                prev_score, _ = dp[i - 1][k]
                if prev_score > best_score:
                    best_score, best_prev = prev_score, k
            if best_prev == -1:
                # No valid predecessor keeps monotonicity: still allow this
                # frame alone so a full-degenerate-only case doesn't crash;
                # the aggregate score naturally loses to better paths.
                row.append(((c.score or 0.0), -2))
            else:
                row.append((best_score + (c.score or 0.0), best_prev))
        dp.append(row)

    last_row = dp[-1]
    end_j = max(range(len(last_row)), key=lambda j: last_row[j][0])
    path = [end_j]
    for i in range(len(trimmed) - 1, 0, -1):
        prev_j = dp[i][path[-1]][1]
        if prev_j < 0:
            return None  # a monotonic path does not exist end-to-end
        path.append(prev_j)
    path.reverse()
    return [trimmed[i][j] for i, j in enumerate(path)]


def align_trake(pools: list[EventPool], strategy: str = "dp") -> list[SearchCandidate] | None:
    if strategy == "greedy":
        return align_greedy(pools)
    result = align_dp(pools)
    if result is None:
        return align_greedy(pools)
    return result


def to_hypothesis(
    query_id: str,
    video_id: str,
    aligned: list[SearchCandidate],
    preprocess_run_id: str,
    *,
    rank: int = 1,
    event_alternatives: tuple[tuple[SearchCandidate, ...], ...] = (),
    verification: dict[str, object] | None = None,
) -> TrakeHypothesis:
    scores = [c.score or 0.0 for c in aligned]
    return TrakeHypothesis(
        query_id=query_id,
        video_id=video_id,
        frame_ids=tuple(c.frame_id for c in aligned),
        event_scores=tuple(scores),
        aggregate_score=sum(scores),
        preprocess_run_id=preprocess_run_id,
        timestamps_ms=tuple(c.timestamp_ms for c in aligned),
        candidates=tuple(aligned),
        rank=rank,
        event_alternatives=event_alternatives,
        verification=verification or {},
    )
