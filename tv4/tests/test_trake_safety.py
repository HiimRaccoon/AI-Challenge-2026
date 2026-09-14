from tv4.contracts import SearchCandidate
from tv4.kis_pipeline import KisServices
from tv4.trake_pipeline import _align_visual_sequence, run_trake_query


def _candidate(frame_id: int, timestamp_ms: int) -> SearchCandidate:
    return SearchCandidate(
        query_id="q", video_id="L24_V019", frame_id=frame_id,
        timestamp_ms=timestamp_ms, source="visual", rank=1, score=0.9,
    )


class _NoTextEvidence:
    def search_all(self, request, routes):
        return {}


class _NoEventCoverageVisual:
    def search_batch(self, requests):
        return [
            [_candidate(100, 1_000)],  # Whole query identifies L24_V019.
            [],                         # But event 1 is visually unsupported.
            [_candidate(200, 2_000)],
        ]


def test_trake_fails_closed_when_visual_events_do_not_cover_identified_video():
    services = KisServices(
        tv1=None, tv3=_NoTextEvidence(), visual=_NoEventCoverageVisual(),
        preprocess_run_id="run_v1_batch1",
    )

    result = run_trake_query(
        "lion dance", services, events=["spin", "land"], query_id="no-coverage"
    )

    assert result is None


def test_trake_selects_complete_event_path_over_uncovered_top_whole_query_video():
    top_whole = SearchCandidate(
        query_id="q", video_id="L24_V045", frame_id=1, timestamp_ms=1_000,
        source="visual", rank=1, score=0.9,
    )
    supported_whole = SearchCandidate(
        query_id="q", video_id="L24_V018", frame_id=2, timestamp_ms=2_000,
        source="visual", rank=2, score=0.8,
    )
    event_one = SearchCandidate(
        query_id="q", video_id="L24_V018", frame_id=3, timestamp_ms=3_000,
        source="visual", rank=1, score=0.9,
    )
    event_two = SearchCandidate(
        query_id="q", video_id="L24_V018", frame_id=4, timestamp_ms=4_000,
        source="visual", rank=1, score=0.9,
    )

    selected = _align_visual_sequence(
        [top_whole, supported_whole], [[event_one], [event_two]], strategy="dp"
    )

    assert selected is not None
    assert selected[0] == "L24_V018"
    assert [candidate.frame_id for candidate in selected[1]] == [3, 4]
