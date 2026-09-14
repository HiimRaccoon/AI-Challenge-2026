"""Structured, deterministic telemetry events with secret redaction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Literal

SECRET_PATTERNS = [
    (re.compile(r"(?i)(password|secret|token|auth|key|authorization)[\s:=]+([^\s,;]+)"), r"\1 [REDACTED]"),
    (re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]+"), "Bearer [REDACTED]"),
]

def redact_secrets(val: Any) -> Any:
    """Recursively redact sensitive keys and pattern matches."""
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            k_lower = str(k).lower()
            if any(s in k_lower for s in ("password", "secret", "token", "api_key", "authorization", "auth_header")):
                out[k] = "[REDACTED]"
            else:
                out[k] = redact_secrets(v)
        return out
    elif isinstance(val, (list, tuple)):
        return [redact_secrets(v) for v in val]
    elif isinstance(val, str):
        cleaned = val
        for pat, repl in SECRET_PATTERNS:
            cleaned = pat.sub(repl, cleaned)
        return cleaned
    return val


TaskMode = Literal["KIS", "VQA", "TRAKE", "FEEDBACK", "OPERATIONS"]
EventType = Literal[
    "QUERY_REQUEST",
    "QUERY_RESPONSE",
    "OPERATOR_INSPECT",
    "OPERATOR_STEP",
    "OPERATOR_REFINE",
    "OPERATOR_LOCK",
    "OPERATOR_APPROVE",
    "BASKET_ADD",
    "BASKET_REMOVE",
    "BASKET_REORDER",
    "EXPORT_CSV",
    "PACKAGE_SUBMISSION",
    "VALIDATION_ERROR",
    "SYSTEM_ERROR",
]


@dataclass(frozen=True)
class TelemetryEvent:
    event_id: str
    event_type: EventType
    task_mode: TaskMode
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query_id: str | None = None
    run_id: str | None = None
    config_id: str | None = None
    model_provenance: dict[str, Any] = field(default_factory=dict)
    source_provenance: dict[str, Any] = field(default_factory=dict)
    branch_degraded_state: dict[str, str] = field(default_factory=dict)
    request_latency_ms: float | None = None
    candidate_count: int | None = None
    result_count: int | None = None
    correction_count: int = 0
    time_first_correct_ms: float | None = None
    validation_errors: tuple[str, ...] = ()
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return redact_secrets(raw)
