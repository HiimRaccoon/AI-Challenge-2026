from wp04.benchmark import BenchmarkRunner
from wp04.contracts import SearchCandidate


def test_benchmark_report_contains_hitk_latency_and_run_ids():
    candidate = SearchCandidate("q", "v", 42, 1400, "ocr", 1, "tv1")
    report = BenchmarkRunner("tv1", "wp04").run({"q": [candidate]}, {"q": {("v", 42)}})
    assert {"hit_at_1", "hit_at_5", "hit_at_20", "latency_ms", "preprocess_run_id", "wp04_artifact_set_id"} <= set(report)
    assert report["hit_at_1"] == 1.0
