"""Held-out exact-frame ON/OFF metric calculator (not a release gate)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LabeledResult:
    query_id: str
    selected_timestamp_ms: int
    interval_start_ms: int
    interval_end_ms: int
    latency_ms: float

    @property
    def hit(self) -> bool:
        return self.interval_start_ms <= self.selected_timestamp_ms <= self.interval_end_ms


@dataclass(frozen=True)
class RunMetrics:
    interval_hit_at_k: float
    p50_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class RefinementBenchmarkReport:
    run_id: str
    config_id: str
    baseline: RunMetrics
    refined: RunMetrics
    interval_hit_delta: float


def metrics(results: Sequence[LabeledResult]) -> RunMetrics:
    if not results:
        raise ValueError("benchmark results cannot be empty")
    latencies = sorted(float(item.latency_ms) for item in results)
    # Inclusive quantiles produce deterministic finite-sample p95 (e.g. 20,50 -> 50).
    return RunMetrics(sum(item.hit for item in results) / len(results), _percentile(latencies, 0.5), _percentile(latencies, 0.95))


def compare_refinement_runs(baseline: Sequence[LabeledResult], refined: Sequence[LabeledResult], *, run_id: str, config_id: str) -> RefinementBenchmarkReport:
    baseline_metrics = metrics(baseline)
    refined_metrics = metrics(refined)
    return RefinementBenchmarkReport(run_id, config_id, baseline_metrics, refined_metrics, refined_metrics.interval_hit_at_k - baseline_metrics.interval_hit_at_k)


def _percentile(values: Sequence[float], percentile: float) -> float:
    # Nearest-rank keeps reported tail latency an observed latency rather than
    # an interpolated value that no request actually incurred.
    index = max(0, min(len(values) - 1, int(-(-len(values) * percentile // 1)) - 1))
    return values[index]
