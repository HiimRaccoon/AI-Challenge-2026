"""Telemetry report generation and summary calculations."""
from __future__ import annotations

from typing import Any, Sequence
from .events import TelemetryEvent


def generate_telemetry_summary(events: Sequence[TelemetryEvent]) -> dict[str, Any]:
    """Calculate latency metrics, error breakdown, and time-first-correct stats."""
    total_events = len(events)
    latencies: list[float] = []
    first_correct_times: list[float] = []
    tasks_count: dict[str, int] = {}
    error_count = 0
    validation_error_count = 0
    degraded_count = 0

    for ev in events:
        tasks_count[ev.task_mode] = tasks_count.get(ev.task_mode, 0) + 1
        if ev.request_latency_ms is not None:
            latencies.append(ev.request_latency_ms)
        if ev.time_first_correct_ms is not None:
            first_correct_times.append(ev.time_first_correct_ms)
        if ev.event_type in ("SYSTEM_ERROR", "VALIDATION_ERROR") or ev.failure_reason:
            error_count += 1
        if ev.validation_errors:
            validation_error_count += len(ev.validation_errors)
        if ev.branch_degraded_state:
            degraded_count += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p90 = latencies[int(len(latencies) * 0.9)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    avg_first_correct = (sum(first_correct_times) / len(first_correct_times)) if first_correct_times else 0.0

    return {
        "total_events": total_events,
        "tasks_breakdown": tasks_count,
        "latency_ms": {
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p99": round(p99, 2),
            "avg": round(avg_latency, 2),
            "samples": len(latencies),
        },
        "time_first_correct_ms": {
            "avg": round(avg_first_correct, 2),
            "samples": len(first_correct_times),
        },
        "error_count": error_count,
        "validation_error_count": validation_error_count,
        "degraded_events_count": degraded_count,
    }
