from __future__ import annotations

from tv4.contracts import SearchCandidate
from tv4.trake_kbest import align_k_best, event_alternatives
from tv4.wp12_trake import EventPool


def _candidate(frame_id: int, timestamp_ms: int, score: float) -> SearchCandidate:
    return SearchCandidate(
        query_id="trake-kbest", video_id="V1", frame_id=frame_id,
        timestamp_ms=timestamp_ms, source="fusion", rank=1, score=score,
    )


def test_align_k_best_returns_distinct_complete_strictly_ordered_sequences():
    pools = (
        EventPool(0, (_candidate(10, 100, 0.9), _candidate(11, 120, 0.8))),
        EventPool(1, (_candidate(20, 200, 0.9), _candidate(21, 230, 0.7))),
        EventPool(2, (_candidate(30, 300, 0.9), _candidate(31, 330, 0.6))),
    )

    paths = align_k_best(pools, k=5)

    assert len(paths) == 5
    assert [item.frame_id for item in paths[0].candidates] == [10, 20, 30]
    assert len({path.identity for path in paths}) == len(paths)
    assert all(
        all(left.timestamp_ms < right.timestamp_ms for left, right in zip(path.candidates, path.candidates[1:]))
        for path in paths
    )


def test_event_alternatives_respect_selected_neighbor_bounds():
    selected = (
        _candidate(10, 100, 0.9),
        _candidate(20, 200, 0.9),
        _candidate(30, 300, 0.9),
    )
    pools = (
        EventPool(0, (selected[0], _candidate(12, 150, 0.8), _candidate(13, 250, 0.95))),
        EventPool(1, (selected[1], _candidate(21, 180, 0.7), _candidate(22, 280, 0.8), _candidate(23, 320, 0.99))),
        EventPool(2, (selected[2], _candidate(31, 310, 0.8), _candidate(29, 190, 0.95))),
    )

    alternatives = event_alternatives(pools, selected)

    assert [item.frame_id for item in alternatives[0]] == [12]
    assert [item.frame_id for item in alternatives[1]] == [22, 21]
    assert [item.frame_id for item in alternatives[2]] == [31]
