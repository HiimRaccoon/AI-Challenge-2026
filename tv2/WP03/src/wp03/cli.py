"""Operational CLI for validating, building, inspecting and searching WP03."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import yaml

from .artifacts import sha256_file
from .config import RuntimeProfile, load_build_config
from .contracts import ContractError, SearchRequest
from .corpus import load_corpus
from .digests import ContentValidationMode
from .model_lock import create_model_lock
from .orchestrator import BuildRequest, build_all_models
from .search import SearchEncoder, search_visual, search_visual_batch
from .worker_launcher import WorkerProcessEncoder
from .worker_protocol import compatibility_fingerprint
from .workers.beit3 import Adapter as Beit3Adapter
from .workers.beit3 import BEIT3_EXPECTED_SIZE_BYTES


SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True)
class CliServices:
    """Injection seam used by CPU tests; normal CLI does not load model code in-process."""

    encoders: Mapping[str, SearchEncoder] = field(default_factory=dict)


def validate_run_id(run_id: str) -> str:
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ContractError("run_id must be a safe filename component")
    return run_id


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _select_online_manifests(paths: Sequence[Path], raw_models: str | None) -> list[Path]:
    """Select an explicit online model subset without altering build artifacts."""
    available = {path.stem: path for path in paths}
    if not raw_models:
        return [available[key] for key in sorted(available)]
    requested = tuple(dict.fromkeys(
        value.strip() for value in raw_models.split(",") if value.strip()
    ))
    if not requested or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in requested):
        raise ContractError("models must be a comma-separated list of safe model keys")
    missing = [value for value in requested if value not in available]
    if missing:
        raise ContractError(f"requested online model manifest is absent: {', '.join(missing)}")
    return [available[value] for value in requested]


def _candidate_frame_filter(raw_json: str | None) -> dict[str, object]:
    if not raw_json:
        return {}
    try:
        values = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ContractError("candidate_frames_json must be valid JSON") from exc
    if not isinstance(values, list) or len(values) > 2_000:
        raise ContractError("candidate_frames_json must contain at most 2000 frame objects")
    frames: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ContractError("candidate_frames_json must contain frame objects")
        video_id, frame_id = value.get("video_id"), value.get("frame_id")
        if (
            not isinstance(video_id, str) or not video_id
            or isinstance(frame_id, bool) or not isinstance(frame_id, int)
            or frame_id < 0
        ):
            raise ContractError("candidate frame requires video_id and non-negative frame_id")
        frames.append({"video_id": video_id, "frame_id": frame_id})
    return {"candidate_frames": frames} if frames else {}


def _load_config(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError("config cannot be read") from exc
    if not isinstance(value, dict):
        raise ContractError("config must be a mapping")
    return value


def _validate(args: argparse.Namespace) -> dict[str, object]:
    corpus = load_corpus(Path(args.data_root), PurePosixPath(args.frames), None)
    return {"status": "ok", "records": len(corpus), "preprocess_run_id": corpus[0].preprocess_run_id}


def _build(args: argparse.Namespace, services: CliServices) -> dict[str, object]:
    run_id = validate_run_id(args.run_id)
    config = _load_config(Path(args.config))
    profile = RuntimeProfile.load(Path(args.runtime_profile), Path(args.runtime_root))
    mode = ContentValidationMode(args.content_validation)
    corpus = load_corpus(Path(args.data_root), PurePosixPath(args.frames), None)
    selected = config.get("selected_videos")
    if selected is not None:
        if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
            raise ContractError("selected_videos must be null or strings")
        corpus = tuple(record for record in corpus if record.video_id in selected)
    if not corpus:
        raise ContractError("selected corpus is empty")
    rows_per_shard = config.get("rows_per_shard")
    if not isinstance(rows_per_shard, int) or rows_per_shard <= 0:
        raise ContractError("rows_per_shard must be positive")
    if mode is ContentValidationMode.STRICT:
        # Strict digest verification occurs when the build orchestrator is invoked.
        pass
    build_config = load_build_config(config)
    requests: list[BuildRequest] = []
    encoders: dict[str, object] = {}
    config_digest = sha256_file(Path(args.config))
    frames_digest = sha256_file(Path(args.data_root) / args.frames)
    for model_key, spec in build_config.models.items():
        fingerprint = compatibility_fingerprint(spec=spec)
        encoder = services.encoders.get(model_key)
        if encoder is None:
            encoder = WorkerProcessEncoder(
                command=profile.command_for(model_key),
                job_root=Path(args.artifact_root) / "jobs" / model_key,
                model_key=model_key,
                revision=spec.revision,
                device="cuda",
                dtype=spec.dtype,
                batch_size=spec.batch_size,
                timeout_seconds=spec.timeout_seconds,
                fallback_dtype=spec.fallback_dtype,
                expected_dimension=spec.expected_dimension,
                _compatibility_fingerprint=fingerprint,
                environment=profile.environment_for(model_key),
            )
        if not hasattr(encoder, "encode_images"):
            raise ContractError(f"configured encoder for {model_key} cannot encode images")
        encoders[model_key] = encoder
        requests.append(
            BuildRequest(
                run_id, model_key, spec.revision, Path(args.data_root),
                Path(args.artifact_root), corpus, rows_per_shard, resume=args.resume,
                content_validation=mode, config_digest=config_digest, frames_input_digest=frames_digest,
                code_version=args.code_version, compatibility_fingerprint=fingerprint,
                expected_dimension=spec.expected_dimension,
                rrf_k=build_config.rrf_k, dedup_window_ms=build_config.dedup_window_ms,
                digest_workers=build_config.digest_workers,
                index_config=build_config.index,
            )
        )
    return build_all_models(requests, encoders).to_dict()


def _search(args: argparse.Namespace, services: CliServices) -> dict[str, object]:
    if args.top_k <= 0:
        raise ContractError("top_k must be positive")
    if not args.query.strip():
        raise ContractError("query must not be empty")
    encoders: Mapping[str, SearchEncoder] = services.encoders
    completed_manifests = _select_online_manifests(
        sorted((Path(args.artifact_root) / "manifests").glob("*.json")),
        getattr(args, "models", None),
    )
    policy = next(
        (
            json.loads(path.read_text(encoding="utf-8"))
            for path in completed_manifests
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
        ),
        {},
    )
    rrf_k = args.rrf_k if args.rrf_k is not None else int(policy.get("rrf_k", 60))
    dedup_window_ms = args.dedup_window_ms if args.dedup_window_ms is not None else policy.get("dedup_window_ms", 1_000)
    if not encoders:
        if not args.runtime_root or not args.runtime_profile:
            raise ContractError("search requires runtime root/profile unless test encoders are injected")
        profile = RuntimeProfile.load(Path(args.runtime_profile), Path(args.runtime_root))
        encoders = {}
        for manifest_path in completed_manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_key = str(manifest.get("model_key", ""))
            if manifest.get("status") == "complete" and model_key:
                encoders[model_key] = WorkerProcessEncoder(
                    command=profile.command_for(model_key),
                    job_root=Path(args.artifact_root) / "jobs" / model_key,
                    model_key=model_key, revision=str(manifest.get("model_version", "")),
                    device="cuda", dtype="float16", batch_size=1,
                    _compatibility_fingerprint=str(manifest.get("compatibility_fingerprint", "")),
                    environment=profile.environment_for(model_key),
                )
        if not encoders:
            raise ContractError("no complete model manifest is available")
    request = SearchRequest(
        query_id=args.query_id, task="KIS", query_text=args.query, question=None,
        events=(), filters=_candidate_frame_filter(
            getattr(args, "candidate_frames_json", None),
        ), limit=args.top_k, language=None, session_id=None,
        event_index=args.event_index, internal_pool=args.top_k > 100,
    )
    response = search_visual(
        request=request,
        artifact_root=Path(args.artifact_root),
        encoders=dict(encoders),
        candidate_k_per_model=args.candidate_k_per_model,
        hard_candidate_cap=args.hard_candidate_cap,
        rrf_k=rrf_k,
        dedup_window_ms=dedup_window_ms,
    )
    return response.to_dict()


def _search_batch(args: argparse.Namespace, services: CliServices) -> dict[str, object]:
    """Public multi-query counterpart of ``search`` for TRAKE workloads."""
    if args.top_k <= 0:
        raise ContractError("top_k must be positive")
    try:
        raw_queries = json.loads(args.queries_json)
    except json.JSONDecodeError as exc:
        raise ContractError("queries_json must be valid JSON") from exc
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ContractError("queries_json must be a non-empty list")
    requests: list[SearchRequest] = []
    candidate_filter = _candidate_frame_filter(
        getattr(args, "candidate_frames_json", None),
    )
    for row in raw_queries:
        if not isinstance(row, dict):
            raise ContractError("each batch query must be an object")
        query_id, query_text, event_index = row.get("query_id"), row.get("query_text"), row.get("event_index")
        if not isinstance(query_id, str) or not query_id.strip() or not isinstance(query_text, str) or not query_text.strip():
            raise ContractError("each batch query requires non-empty query_id and query_text")
        if event_index is not None and (isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0):
            raise ContractError("event_index must be a non-negative integer or null")
        requests.append(SearchRequest(
            query_id=query_id,
            task="TRAKE",
            query_text=query_text,
            question=None,
            events=(),
            filters=candidate_filter,
            limit=args.top_k,
            language=None,
            session_id=None,
            event_index=event_index,
            internal_pool=args.top_k > 100,
        ))

    encoders: Mapping[str, SearchEncoder] = services.encoders
    completed_manifests = _select_online_manifests(
        sorted((Path(args.artifact_root) / "manifests").glob("*.json")),
        getattr(args, "models", None),
    )
    policy = next(
        (
            json.loads(path.read_text(encoding="utf-8"))
            for path in completed_manifests
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
        ),
        {},
    )
    rrf_k = args.rrf_k if args.rrf_k is not None else int(policy.get("rrf_k", 60))
    dedup_window_ms = args.dedup_window_ms if args.dedup_window_ms is not None else policy.get("dedup_window_ms", 1_000)
    if not encoders:
        if not args.runtime_root or not args.runtime_profile:
            raise ContractError("search-batch requires runtime root/profile unless test encoders are injected")
        profile = RuntimeProfile.load(Path(args.runtime_profile), Path(args.runtime_root))
        encoders = {}
        for manifest_path in completed_manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model_key = str(manifest.get("model_key", ""))
            if manifest.get("status") == "complete" and model_key:
                encoders[model_key] = WorkerProcessEncoder(
                    command=profile.command_for(model_key),
                    job_root=Path(args.artifact_root) / "jobs" / model_key,
                    model_key=model_key, revision=str(manifest.get("model_version", "")),
                    device="cuda", dtype="float16", batch_size=max(1, len(requests)),
                    _compatibility_fingerprint=str(manifest.get("compatibility_fingerprint", "")),
                    environment=profile.environment_for(model_key),
                )
        if not encoders:
            raise ContractError("no complete model manifest is available")

    responses = search_visual_batch(
        requests=tuple(requests), artifact_root=Path(args.artifact_root), encoders=dict(encoders),
        candidate_k_per_model=args.candidate_k_per_model, hard_candidate_cap=args.hard_candidate_cap,
        rrf_k=rrf_k, dedup_window_ms=dedup_window_ms,
    )
    return {"schema_version": "1.0.0", "results": [response.to_dict() for response in responses]}


def _inspect(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.artifact_root)
    manifests = sorted((root / "manifests").glob("*.json"))
    if not manifests:
        raise ContractError("no manifests found")
    results: dict[str, object] = {}
    for path in manifests:
        body = json.loads(path.read_text(encoding="utf-8"))
        model_key = str(body.get("model_key", ""))
        if body.get("status") != "complete" or not model_key:
            raise ContractError(f"manifest {path.name} is not complete")
        for kind, suffix, field in (("indexes", ".faiss", "index_sha256"), ("embedding_maps", ".parquet", "mapping_sha256")):
            artifact = root / kind / f"{model_key}{suffix}"
            if sha256_file(artifact) != body.get(field):
                raise ContractError(f"{model_key} {kind} digest mismatch")
        results[model_key] = {"vector_count": body.get("vector_count"), "status": "complete"}
    return {"status": "ok", "models": results}


def _serve_stdio(args: argparse.Namespace, services: CliServices) -> int:
    """Serve newline-delimited JSON while retaining CPU indexes in memory.

    Model workers are still invoked sequentially, which bounds GPU residency.
    Only the coordinator and immutable FAISS/mapping cache stay alive.
    """
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ContractError("stdio request must be an object")
            operation = request.get("operation")
            common = {
                **vars(args),
                "top_k": int(request.get("top_k", args.top_k)),
                "candidate_k_per_model": int(request.get(
                    "candidate_k_per_model", args.candidate_k_per_model,
                )),
                "models": request.get("models", args.models),
                "candidate_frames_json": json.dumps(
                    request.get("candidate_frames", []), ensure_ascii=False,
                ),
            }
            if operation == "search":
                query = request.get("query_text")
                query_id = request.get("query_id")
                if not isinstance(query, str) or not isinstance(query_id, str):
                    raise ContractError("search requires query_id and query_text")
                result = _search(argparse.Namespace(
                    **common, query=query, query_id=query_id,
                    event_index=request.get("event_index"),
                ), services)
            elif operation == "search-batch":
                queries = request.get("queries")
                result = _search_batch(argparse.Namespace(
                    **common, queries_json=json.dumps(queries, ensure_ascii=False),
                ), services)
            else:
                raise ContractError("operation must be search or search-batch")
            response: object = result
        except (ContractError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            response = {"status": "error", "error": str(exc)}
        print(json.dumps(response, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _lock_model(args: argparse.Namespace) -> dict[str, object]:
    if args.model != "beit3":
        raise ContractError("only beit3 has a checkpoint lock command")
    lock = create_model_lock(
        Path(args.checkpoint), Path(args.lock_path), Beit3Adapter.identity(), BEIT3_EXPECTED_SIZE_BYTES
    )
    return {"status": "locked", "model": lock.identity.name, "sha256": lock.sha256, "lock_path": str(lock.path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wp03")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--data-root", required=True)
    validate.add_argument("--frames", required=True)
    build = commands.add_parser("build")
    build.add_argument("--data-root", required=True)
    build.add_argument("--frames", required=True)
    build.add_argument("--run-id", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--runtime-root", required=True)
    build.add_argument("--runtime-profile", required=True)
    build.add_argument("--content-validation", choices=[item.value for item in ContentValidationMode], required=True)
    build.add_argument("--artifact-root", default="artifacts")
    build.add_argument("--code-version", required=True)
    build.add_argument("--resume", action="store_true")
    search = commands.add_parser("search")
    search.add_argument("--artifact-root", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--query-id", default="wp03-query")
    search.add_argument("--event-index", type=int)
    search.add_argument("--top-k", type=int, required=True)
    search.add_argument("--candidate-k-per-model", type=int, default=200)
    search.add_argument("--hard-candidate-cap", type=int)
    search.add_argument("--rrf-k", type=int)
    search.add_argument("--dedup-window-ms", type=int)
    search.add_argument("--runtime-root")
    search.add_argument("--runtime-profile")
    search.add_argument("--models")
    search.add_argument("--candidate-frames-json")
    search_batch = commands.add_parser("search-batch")
    search_batch.add_argument("--artifact-root", required=True)
    search_batch.add_argument("--queries-json", required=True)
    search_batch.add_argument("--top-k", type=int, required=True)
    search_batch.add_argument("--candidate-k-per-model", type=int, default=200)
    search_batch.add_argument("--hard-candidate-cap", type=int)
    search_batch.add_argument("--rrf-k", type=int)
    search_batch.add_argument("--dedup-window-ms", type=int)
    search_batch.add_argument("--runtime-root")
    search_batch.add_argument("--runtime-profile")
    search_batch.add_argument("--models")
    search_batch.add_argument("--candidate-frames-json")
    serve = commands.add_parser("serve-stdio")
    serve.add_argument("--artifact-root", required=True)
    serve.add_argument("--top-k", type=int, default=100)
    serve.add_argument("--candidate-k-per-model", type=int, default=200)
    serve.add_argument("--hard-candidate-cap", type=int)
    serve.add_argument("--rrf-k", type=int)
    serve.add_argument("--dedup-window-ms", type=int)
    serve.add_argument("--runtime-root")
    serve.add_argument("--runtime-profile")
    serve.add_argument("--models")
    serve.add_argument("--candidate-frames-json")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--artifact-root", required=True)
    lock = commands.add_parser("lock-model")
    lock.add_argument("--model", required=True)
    lock.add_argument("--checkpoint", required=True)
    lock.add_argument("--lock-path", default="model-locks/beit3.json")
    return parser


def main(argv: Sequence[str] | None = None, services: CliServices | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        selected = services or CliServices()
        if args.command == "validate":
            result = _validate(args)
        elif args.command == "build":
            result = _build(args, selected)
        elif args.command == "search":
            result = _search(args, selected)
        elif args.command == "search-batch":
            result = _search_batch(args, selected)
        elif args.command == "serve-stdio":
            return _serve_stdio(args, selected)
        elif args.command == "inspect":
            result = _inspect(args)
        else:
            result = _lock_model(args)
        _print_json(result)
        if args.command == "build" and isinstance(result, dict) and result.get("status") == "failed":
            return 2
        return 0
    except (ContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
