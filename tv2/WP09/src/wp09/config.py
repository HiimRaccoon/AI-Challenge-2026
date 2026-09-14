"""Configuration loading for reproducible WP09 refinement runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import ContractError


@dataclass(frozen=True)
class RefinementConfig:
    default_radius_ms: int
    coarse_sample_fps: float
    dense_radius_ms: int
    dense_sample_fps: float
    hypothesis_limit: int
    refiner_model: str
    batch_size: int
    failure_mode: str
    min_radius_ms: int | None = None
    max_radius_ms: int | None = None
    dense_seed_count: int = 1
    stable_variance_penalty: float = 0.25
    decoder_config: str = "pyav-original-pts-v1"
    config_version: str = "wp09-v1"
    cache_max_entries: int = 32
    cache_ttl_seconds: float = 300.0

    def radius_for_confidence(self, confidence: float | None, budget: "DecodeBudget") -> int:
        """Lower upstream confidence broadens search, never beyond the request budget."""
        from .contracts import DecodeBudget
        if not isinstance(budget, DecodeBudget):
            raise ContractError("decode budget is required")
        low = self.min_radius_ms if self.min_radius_ms is not None else max(1, self.default_radius_ms // 2)
        high = self.max_radius_ms if self.max_radius_ms is not None else self.default_radius_ms * 2
        confidence_value = 0.5 if confidence is None else confidence
        radius = int(low + (high - low) * (1.0 - confidence_value))
        return min(max(low, radius), high, budget.max_window_ms // 2)

    @classmethod
    def load(cls, path: Path) -> "RefinementConfig":
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ContractError("refinement config cannot be read") from exc
        if not isinstance(raw, Mapping) or not isinstance(raw.get("wp09"), Mapping):
            raise ContractError("refinement config must contain a wp09 mapping")
        values = raw["wp09"]
        return cls(
            default_radius_ms=_positive_int(values, "default_radius_ms"),
            coarse_sample_fps=_positive_float(values, "coarse_sample_fps"),
            dense_radius_ms=_positive_int(values, "dense_radius_ms"),
            dense_sample_fps=_positive_float(values, "dense_sample_fps"),
            hypothesis_limit=_positive_int(values, "hypothesis_limit"),
            refiner_model=_non_empty_string(values, "refiner_model"),
            batch_size=_positive_int(values, "batch_size"),
            failure_mode=_failure_mode(values),
            min_radius_ms=_optional_positive_int(values, "min_radius_ms"),
            max_radius_ms=_optional_positive_int(values, "max_radius_ms"),
            dense_seed_count=_optional_positive_int(values, "dense_seed_count") or 1,
            stable_variance_penalty=_optional_non_negative_float(values, "stable_variance_penalty", 0.25),
            decoder_config=_optional_string(values, "decoder_config", "pyav-original-pts-v1"),
            config_version=_optional_string(values, "config_version", "wp09-v1"),
            cache_max_entries=_optional_positive_int(values, "cache_max_entries") or 32,
            cache_ttl_seconds=_optional_positive_float(values, "cache_ttl_seconds", 300.0),
        )


def _positive_int(values: Mapping[str, Any], field: str) -> int:
    value = values.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _positive_float(values: Mapping[str, Any], field: str) -> float:
    value = values.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ContractError(f"{field} must be a positive number")
    return float(value)


def _non_empty_string(values: Mapping[str, Any], field: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _failure_mode(values: Mapping[str, Any]) -> str:
    value = _non_empty_string(values, "failure_mode")
    if value != "manual_only":
        raise ContractError("failure_mode must be manual_only")
    return value


def _optional_positive_int(values: Mapping[str, Any], field: str) -> int | None:
    if field not in values:
        return None
    return _positive_int(values, field)


def _optional_non_negative_float(values: Mapping[str, Any], field: str, default: float) -> float:
    if field not in values:
        return default
    value = values[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ContractError(f"{field} must be a non-negative number")
    return float(value)


def _optional_positive_float(values: Mapping[str, Any], field: str, default: float) -> float:
    return default if field not in values else _positive_float(values, field)


def _optional_string(values: Mapping[str, Any], field: str, default: str) -> str:
    return default if field not in values else _non_empty_string(values, field)
