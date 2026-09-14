from __future__ import annotations

import json

from wp09.benchmark import LabeledResult, compare_refinement_runs
from wp09.cache import CachedWindow, DecodedWindowCache, DecodedWindowKey, WindowRequestKey
from wp09.cli import main
from wp09.decoder import DecodedFrame


def _window_key(video_id: str) -> DecodedWindowKey:
    return DecodedWindowKey("run-a", video_id, 10, 20, "pyav-1")


def _request_key(video_id: str) -> WindowRequestKey:
    return WindowRequestKey("run-a", video_id, 15, 5, (10, 1_000, None, 1), "pyav-1")


def _frames(frame_id: int) -> tuple[DecodedFrame, ...]:
    return (DecodedFrame(frame_id, frame_id, "1/1000", frame_id),)


def _cached_window(key: DecodedWindowKey, frame_id: int) -> CachedWindow:
    return CachedWindow(key, _frames(frame_id), 10, 20, frame_id, False)


def test_cache_key_never_crosses_mapping_runs() -> None:
    """Catches reuse of frames resolved under a stale PTS-to-frame mapping."""
    cache = DecodedWindowCache()
    first = DecodedWindowKey("run-a", "L21_V001", 10, 20, "pyav-1")
    other = DecodedWindowKey("run-b", "L21_V001", 10, 20, "pyav-1")
    cache.put(first, (DecodedFrame(7, 10, "1/1000", 10),))
    assert cache.get(other) is None


def test_cache_evicts_least_recently_used_window_and_its_request_alias() -> None:
    """Catches decoded RGB frames remaining reachable after an LRU eviction."""
    now = [0.0]
    cache = DecodedWindowCache(max_entries=2, ttl_seconds=60.0, clock=lambda: now[0])
    first, second, third = _window_key("first"), _window_key("second"), _window_key("third")
    first_request, second_request = _request_key("first"), _request_key("second")
    cache.put_for_request(first_request, _cached_window(first, 1))
    cache.put_for_request(second_request, _cached_window(second, 2))
    assert cache.get(first) == _frames(1)  # first is now most recently used

    cache.put(third, _frames(3))

    assert cache.get(first) == _frames(1)
    assert cache.get(second) is None
    assert cache.get_for_request(second_request) is None
    assert cache.get(third) == _frames(3)


def test_cache_expiry_removes_window_and_its_request_alias() -> None:
    """Catches a stale request index retaining decoded RGB frames past the TTL."""
    now = [0.0]
    cache = DecodedWindowCache(max_entries=2, ttl_seconds=5.0, clock=lambda: now[0])
    key, request = _window_key("one"), _request_key("one")
    cache.put_for_request(request, _cached_window(key, 1))

    now[0] = 5.01

    assert cache.get(key) is None
    assert cache.get_for_request(request) is None


def test_cache_bounds_request_aliases_for_a_hot_canonical_window() -> None:
    """Catches request-index metadata growing forever while one video window stays hot."""
    cache = DecodedWindowCache(max_entries=2, ttl_seconds=60.0, clock=lambda: 0.0)
    key = _window_key("one")
    first = _request_key("first")
    second = _request_key("second")
    third = _request_key("third")
    entry = _cached_window(key, 1)
    cache.put_for_request(first, entry)
    cache.put_for_request(second, entry)
    cache.put_for_request(third, entry)

    assert cache.get_for_request(first) is None
    assert cache.get_for_request(second) is not None
    assert cache.get_for_request(third) is not None
    assert cache.get(key) == _frames(1)


def test_benchmark_calculates_interval_and_latency_deltas() -> None:
    """Catches reporting a refinement comparison without literal ON/OFF metrics."""
    baseline = (LabeledResult("a", 100, 95, 105, 30), LabeledResult("b", 180, 190, 210, 70))
    refined = (LabeledResult("a", 100, 95, 105, 20), LabeledResult("b", 200, 190, 210, 50))
    report = compare_refinement_runs(baseline, refined, run_id="run-7", config_id="siglip2-b16-224")
    assert report.baseline.interval_hit_at_k == 0.5
    assert report.refined.interval_hit_at_k == 1.0
    assert report.interval_hit_delta == 0.5
    assert report.refined.p95_latency_ms == 50


def test_cli_emits_json_error_for_invalid_request(capsys) -> None:
    """Catches integration callers receiving a traceback instead of the JSON contract."""
    assert main(["refine", "--request", "missing-request.json", "--config", "configs/default.yaml"]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "error"
