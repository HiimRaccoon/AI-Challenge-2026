"""Runtime-profile loading without allowing commands to escape runtime_root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import ContractError


@dataclass(frozen=True)
class IndexConfig:
    """FAISS build/search policy shared by every visual model."""

    type: str = "flat_ip"
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 512
    shot_expansion_per_hit: int = 0

    @classmethod
    def from_mapping(cls, raw: object) -> "IndexConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ContractError("index must be a mapping")
        index_type = raw.get("type", "flat_ip")
        hnsw_m = raw.get("hnsw_m", 32)
        ef_construction = raw.get("ef_construction", 200)
        ef_search = raw.get("ef_search", 512)
        shot_expansion_per_hit = raw.get("shot_expansion_per_hit", 0)
        if index_type not in {"flat_ip", "hnsw_flat"}:
            raise ContractError("index.type must be flat_ip or hnsw_flat")
        for field, value in (
            ("index.hnsw_m", hnsw_m),
            ("index.ef_construction", ef_construction),
            ("index.ef_search", ef_search),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(f"{field} must be a positive integer")
        if (
            not isinstance(shot_expansion_per_hit, int)
            or isinstance(shot_expansion_per_hit, bool)
            or shot_expansion_per_hit < 0
        ):
            raise ContractError("index.shot_expansion_per_hit must be a non-negative integer")
        return cls(
            type=index_type,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            shot_expansion_per_hit=shot_expansion_per_hit,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "hnsw_m": self.hnsw_m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "shot_expansion_per_hit": self.shot_expansion_per_hit,
        }


def _required_string(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _required_mapping(raw: Mapping[str, object], field: str) -> dict[str, object]:
    value = raw.get(field)
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class ModelRuntimeSpec:
    """Semantic and operational settings for exactly one isolated model worker."""

    model_key: str
    model_id: str
    revision: str
    tokenizer_revision: str
    image_preprocess: Mapping[str, object]
    text_preprocess: Mapping[str, object]
    query_template: str
    expected_dimension: int
    dtype: str
    fallback_dtype: str | None
    batch_size: int
    timeout_seconds: int

    @classmethod
    def from_mapping(cls, model_key: str, raw: Mapping[str, object]) -> "ModelRuntimeSpec":
        if not model_key:
            raise ContractError("model key must be a non-empty string")
        expected_dimension = raw.get("expected_dimension")
        batch_size = raw.get("batch_size")
        timeout_seconds = raw.get("timeout_seconds")
        fallback_dtype = raw.get("fallback_dtype")
        if not isinstance(expected_dimension, int) or expected_dimension <= 0:
            raise ContractError("expected_dimension must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ContractError("batch_size must be a positive integer")
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise ContractError("timeout_seconds must be a positive integer")
        if fallback_dtype is not None and (not isinstance(fallback_dtype, str) or not fallback_dtype):
            raise ContractError("fallback_dtype must be a non-empty string or null")
        return cls(
            model_key=model_key,
            model_id=_required_string(raw, "model_id"),
            revision=_required_string(raw, "revision"),
            tokenizer_revision=_required_string(raw, "tokenizer_revision"),
            image_preprocess=_required_mapping(raw, "image_preprocess"),
            text_preprocess=_required_mapping(raw, "text_preprocess"),
            query_template=_required_string(raw, "query_template"),
            expected_dimension=expected_dimension,
            dtype=_required_string(raw, "dtype"),
            fallback_dtype=fallback_dtype,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "image_preprocess": dict(self.image_preprocess),
            "text_preprocess": dict(self.text_preprocess),
            "query_template": self.query_template,
            "expected_dimension": self.expected_dimension,
            "normalization": "l2_float32",
        }


@dataclass(frozen=True)
class BuildConfig:
    models: Mapping[str, ModelRuntimeSpec]
    rrf_k: int
    dedup_window_ms: int | None
    digest_workers: int
    index: IndexConfig


def load_build_config(raw: Mapping[str, object]) -> BuildConfig:
    """Parse model semantics once, before a build starts any worker process."""

    raw_models = raw.get("models")
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise ContractError("config requires at least one model")
    models: dict[str, ModelRuntimeSpec] = {}
    for model_key, settings in raw_models.items():
        if not isinstance(model_key, str) or not isinstance(settings, Mapping):
            raise ContractError("model config is invalid")
        models[model_key] = ModelRuntimeSpec.from_mapping(model_key, settings)
    rrf_k = raw.get("rrf_k", 60)
    dedup_window_ms = raw.get("dedup_window_ms", None)
    digest_workers = raw.get("digest_workers", 1)
    if not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ContractError("rrf_k must be a positive integer")
    if dedup_window_ms is not None and (not isinstance(dedup_window_ms, int) or dedup_window_ms < 0):
        raise ContractError("dedup_window_ms must be a non-negative integer or null")
    if not isinstance(digest_workers, int) or isinstance(digest_workers, bool) or digest_workers <= 0:
        raise ContractError("digest_workers must be a positive integer")
    return BuildConfig(
        models=models,
        rrf_k=rrf_k,
        dedup_window_ms=dedup_window_ms,
        digest_workers=digest_workers,
        index=IndexConfig.from_mapping(raw.get("index")),
    )


@dataclass(frozen=True)
class RuntimeProfile:
    runtime_root: Path
    workers: dict[str, tuple[str, ...]]
    environments: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: Path, runtime_root: Path) -> "RuntimeProfile":
        try:
            raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ContractError("runtime profile cannot be read") from exc
        root = runtime_root.resolve()
        workers = raw.get("workers")
        if not isinstance(workers, dict):
            raise ContractError("runtime profile requires workers")
        parsed: dict[str, tuple[str, ...]] = {}
        environments: dict[str, dict[str, str]] = {}
        for model_key, settings in workers.items():
            if not isinstance(model_key, str):
                raise ContractError("runtime worker command is invalid")
            if isinstance(settings, list):
                command = settings
                raw_environment: Mapping[str, object] = {}
            elif isinstance(settings, Mapping):
                command = settings.get("command")
                raw_environment = settings.get("env", {})
            else:
                raise ContractError("runtime worker command is invalid")
            if not isinstance(command, list) or not command or not isinstance(raw_environment, Mapping):
                raise ContractError("runtime worker command is invalid")
            executable = Path(str(command[0]))
            if executable.is_absolute() or executable.drive or str(executable).startswith("\\\\"):
                raise ContractError("runtime worker executable must be relative")
            resolved = (root / executable).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ContractError("runtime worker executable escapes runtime_root") from exc
            parsed[model_key] = (str(resolved), *(str(part) for part in command[1:]))
            environment: dict[str, str] = {}
            for name, raw_value in raw_environment.items():
                if not isinstance(name, str) or not name.startswith("WP03_") or not isinstance(raw_value, str):
                    raise ContractError("runtime worker environment is invalid")
                relative = Path(raw_value)
                if relative.is_absolute() or relative.drive:
                    raise ContractError("runtime worker environment path must be relative")
                value = (root / relative).resolve()
                try:
                    value.relative_to(root)
                except ValueError as exc:
                    raise ContractError("runtime worker environment escapes runtime_root") from exc
                environment[name] = str(value)
            environments[model_key] = environment
        return cls(root, parsed, environments)

    def command_for(self, model_key: str) -> tuple[str, ...]:
        try:
            return self.workers[model_key]
        except KeyError as exc:
            raise ContractError(f"runtime profile has no worker for {model_key}") from exc

    def environment_for(self, model_key: str) -> Mapping[str, str]:
        try:
            return dict(self.environments[model_key])
        except KeyError as exc:
            raise ContractError(f"runtime profile has no worker for {model_key}") from exc
