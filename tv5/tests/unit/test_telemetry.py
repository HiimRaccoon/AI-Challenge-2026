"""Unit tests for structured operational telemetry, secret redaction, and reports."""
from __future__ import annotations

from pathlib import Path
import pytest

from tv5.telemetry import TelemetryEvent, TelemetryRecorder, generate_telemetry_summary, redact_secrets


def test_secret_redaction_removes_sensitive_keys_and_patterns() -> None:
    data = {
        "user": "operator_1",
        "api_key": "secret-12345-token",
        "nested": {
            "password": "super-secret-password",
            "authorization": "Bearer eyJhbGciOi...",
            "notes": "Query text mentioning token=998877",
        },
        "query": "find person in red shirt",
    }
    redacted = redact_secrets(data)
    assert redacted["user"] == "operator_1"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["authorization"] == "[REDACTED]"
    assert "[REDACTED]" in redacted["nested"]["notes"]
    assert redacted["query"] == "find person in red shirt"


def test_telemetry_event_serialization_and_redaction() -> None:
    event = TelemetryEvent(
        event_id="ev-001",
        event_type="QUERY_REQUEST",
        task_mode="KIS",
        query_id="q-100",
        run_id="run_v1_batch1",
        config_id="default_cfg",
        model_provenance={"visual": "metaclip2", "auth_token": "secret_abc"},
        request_latency_ms=124.5,
        candidate_count=100,
    )
    d = event.to_dict()
    assert d["event_id"] == "ev-001"
    assert d["task_mode"] == "KIS"
    assert d["model_provenance"]["visual"] == "metaclip2"
    assert d["model_provenance"]["auth_token"] == "[REDACTED]"


def test_telemetry_recorder_bounded_and_file_output(tmp_path: Path) -> None:
    log_file = tmp_path / "telemetry.jsonl"
    recorder = TelemetryRecorder(max_events=5, output_path=log_file)

    for i in range(10):
        recorder.record(
            TelemetryEvent(
                event_id=f"ev-{i}",
                event_type="QUERY_RESPONSE",
                task_mode="KIS" if i % 2 == 0 else "VQA",
                request_latency_ms=50.0 + i * 10,
                time_first_correct_ms=200.0 + i * 5 if i > 5 else None,
            )
        )

    # Bounded in-memory size
    assert recorder.count() == 5
    events = recorder.get_events()
    assert events[0].event_id == "ev-5"
    assert events[-1].event_id == "ev-9"

    # File output has all 10 lines
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 10

    # Summary calculations
    summary = generate_telemetry_summary(events)
    assert summary["total_events"] == 5
    assert summary["latency_ms"]["samples"] == 5
    assert summary["latency_ms"]["p50"] >= 100.0
