"""Command-line handoff for querying and checking a WP04 artifact set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .adapters import AdapterUnavailableError, load_adapter
from .config import load_config
from .contracts import AudioRecord, FrameRecord, SearchCandidate, WP04RunIdentity
from .fingerprints import build_input_fingerprint
from .pipeline import WP04Pipeline
from .retrieval import LocalTextIndex
from .storage import ArtifactStore
from .tv1_export import TV1ManifestExportError, export_tv1_manifests
from .validation import validate_run


class _UnavailableAdapter:
    def __init__(self, name: str, version: str, reason: str) -> None:
        self.model_name, self.model_version, self.vad_version, self._reason = name, version, "unavailable", reason

    def detect(self, frame: FrameRecord):
        raise AdapterUnavailableError(self._reason)

    def transcribe(self, audio: AudioRecord):
        raise AdapterUnavailableError(self._reason)


def _doctor(arguments: argparse.Namespace) -> int:
    """Report whether enabled command-backed adapters are configured."""
    config = load_config(Path(arguments.config))
    modalities = config["modalities"]
    checks: dict[str, Any] = {}
    production_ready = True
    for name in ("ocr", "asr", "object"):
        settings = modalities.get(name, {})
        enabled = bool(settings.get("enabled", False))
        adapter_settings = settings.get("settings", {})
        command_env = adapter_settings.get("command_env")
        configured_command = os.environ.get(command_env, "").strip() if command_env else ""
        server_url = str(adapter_settings.get("server_url", "")).strip()
        transport = "http" if server_url else "command"
        configured = bool(server_url or configured_command)
        status = "disabled" if not enabled else ("ok" if configured else "missing_transport")
        if enabled and not configured:
            production_ready = False
        checks[name] = {
            "enabled": enabled,
            "transport": transport,
            "server_url": server_url or None,
            "command_env": command_env,
            "configured": configured,
            "status": status,
        }
    print(json.dumps({"production_ready": production_ready, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if production_ready else 2


def _text_for_row(source: str, row: dict[str, Any]) -> str:
    if source in {"ocr", "asr"}:
        return str(row.get("raw_text", ""))
    return json.dumps(row.get("fields", {}), ensure_ascii=False, sort_keys=True)


def _reference_for_row(source: str, row: dict[str, Any]) -> str:
    key = {"ocr": "evidence_id", "asr": "segment_id", "object": "evidence_id", "metadata": "record_id"}[source]
    return str(row[key])


def _search(arguments: argparse.Namespace) -> int:
    identity = WP04RunIdentity(arguments.preprocess_run_id, arguments.artifact_set_id, "cli", "cli")
    rows = ArtifactStore(arguments.run_dir, identity).read_all_records(arguments.source)
    documents = [(_reference_for_row(arguments.source, row), _text_for_row(arguments.source, row)) for row in rows]
    hits = LocalTextIndex.from_documents(documents).search(arguments.query, arguments.limit)
    by_reference = {_reference_for_row(arguments.source, row): row for row in rows}
    candidates: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        row = by_reference[hit.document_id]
        if "frame_id" not in row or "timestamp_ms" not in row:
            raise ValueError(f"{arguments.source} requires temporal hypotheses before it can emit SearchCandidate")
        candidates.append(SearchCandidate(
            query_id=arguments.query_id, video_id=row["video_id"], frame_id=int(row["frame_id"]),
            timestamp_ms=int(row["timestamp_ms"]), source=arguments.source, rank=rank,
            preprocess_run_id=arguments.preprocess_run_id, raw_score=hit.score,
            evidence_refs=(_reference_for_row(arguments.source, row),), confidence=row.get("confidence"),
        ).to_dict())
    print(json.dumps({"candidates": candidates}, ensure_ascii=False))
    return 0


def _validate(arguments: argparse.Namespace) -> int:
    frame_rows = json.loads(Path(arguments.frames_json).read_text(encoding="utf-8"))
    if not isinstance(frame_rows, list):
        raise ValueError("frames-json must contain a JSON list")
    frames = [
        FrameRecord(
            row["preprocess_run_id"], row["video_id"], int(row["frame_id"]),
            int(row["keyframe_seq"]), int(row["timestamp_ms"]),
            shot_id=row.get("shot_id"), selection_reason=row.get("selection_reason"),
        )
        for row in frame_rows
    ]
    identity = WP04RunIdentity(arguments.preprocess_run_id, arguments.artifact_set_id, "cli", "cli")
    report = validate_run(Path(arguments.run_dir), identity, frames)
    print(json.dumps(report.to_dict(), ensure_ascii=False))
    return 0 if report.is_valid else 1


def _frames_from_json(path: str) -> list[FrameRecord]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("frames-json must contain a JSON list")
    return [
        FrameRecord(row["preprocess_run_id"], row["video_id"], int(row["frame_id"]),
                    int(row["keyframe_seq"]), int(row["timestamp_ms"]),
                    row.get("keyframe_path"), row.get("keyframe_sha256"),
                    row.get("shot_id"), row.get("selection_reason"))
        for row in rows
    ]


def _audio_from_json(path: str) -> dict[str, AudioRecord]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("audio-json must contain a JSON list")
    records = [AudioRecord(row["preprocess_run_id"], row["video_id"], row.get("audio_path"), row.get("checksum"), bool(row["declared_present"]), row.get("duration_ms")) for row in rows]
    return {record.video_id: record for record in records}


def _configured_adapter(settings: dict[str, Any]) -> object:
    model = settings.get("model", {})
    name, version, factory = str(model.get("name", "unknown")), str(model.get("version", "unknown")), str(model.get("factory", ""))
    if not factory:
        return _UnavailableAdapter(name, version, f"no local adapter factory configured for {name}")
    return load_adapter(factory, **dict(settings.get("settings", {})))


def build_preprocess_fingerprint(
    frames: list[FrameRecord], audio: Mapping[str, AudioRecord], config: Mapping[str, Any],
) -> str:
    missing = [frame.video_id + ":" + str(frame.frame_id) for frame in frames if not frame.keyframe_path or not frame.keyframe_sha256]
    if missing:
        raise ValueError(f"TV1 frame records require keyframe_path and keyframe_sha256: {', '.join(missing)}")
    checksums = {
        "frames": [
            {"video_id": frame.video_id, "frame_id": frame.frame_id, "path": frame.keyframe_path,
             "sha256": frame.keyframe_sha256, "shot_id": frame.shot_id,
             "selection_reason": frame.selection_reason}
            for frame in frames
        ],
        "audio": {key: {"path": value.audio_path, "sha256": value.checksum} for key, value in audio.items()},
    }
    modalities = config.get("modalities", {})
    versions = {
        name: {
            "model": str(settings.get("model", {}).get("version", "n/a")),
            "vad": str(settings.get("vad", {}).get("version", "n/a")),
        }
        for name, settings in modalities.items() if isinstance(settings, dict)
    }
    return build_input_fingerprint(checksums, config, versions, normalization_version=str(config["normalization_version"]))


def _runtime_batch_size(variable: str, configured: object) -> int:
    """Read a performance-only batch override without changing artifact semantics."""
    configured_size = int(configured)
    value = os.environ.get(variable, "").strip()
    if not value:
        return configured_size
    batch_size = int(value)
    if batch_size < 1:
        raise ValueError(f"{variable} must be a positive integer")
    return batch_size


def _preprocess(arguments: argparse.Namespace) -> int:
    config = load_config(Path(arguments.config))
    frames, audio = _frames_from_json(arguments.frames_json), _audio_from_json(arguments.audio_json)
    fingerprint = build_preprocess_fingerprint(frames, audio, config)
    identity = WP04RunIdentity(arguments.preprocess_run_id, arguments.artifact_set_id, fingerprint, fingerprint)
    modalities = config["modalities"]
    ocr_batch_size = _runtime_batch_size("WP04_RUNTIME_OCR_BATCH_SIZE", modalities["ocr"].get("batch_size", 1))
    object_batch_size = _runtime_batch_size("WP04_RUNTIME_OBJECT_BATCH_SIZE", modalities["object"].get("batch_size", 1))
    print(
        f"[WP04] runtime batches: ocr={ocr_batch_size}, object={object_batch_size} "
        "(not part of the artifact fingerprint)",
        file=sys.stderr,
        flush=True,
    )
    pipeline = WP04Pipeline(
        _configured_adapter(modalities["ocr"]), _configured_adapter(modalities["asr"]),
        _configured_adapter(modalities["object"]),
        enabled={name: bool(modalities[name]["enabled"]) for name in ("ocr", "asr", "object", "metadata")},
        ocr_batch_size=ocr_batch_size,
        object_batch_size=object_batch_size,
        ocr_sample_interval_ms=int(modalities["ocr"].get("sample_interval_ms", 0)),
        object_sample_interval_ms=int(modalities["object"].get("sample_interval_ms", 0)),
    )
    result = pipeline.run_and_store(frames, audio, fingerprint, ArtifactStore(Path(arguments.run_dir), identity))
    print(json.dumps({"fingerprint": fingerprint, "statuses": [status.to_dict() for status in result.statuses]}, ensure_ascii=False))
    return 0


def _export_tv1_manifests(arguments: argparse.Namespace) -> int:
    result = export_tv1_manifests(
        Path(arguments.tv1_run_dir), Path(arguments.output_dir), overwrite=arguments.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wp04")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="check WP04 command environment variables before preprocess")
    doctor.add_argument("--config", required=True)
    export = subcommands.add_parser("export-tv1-manifests", help="convert TV1 Parquet artifacts into WP04 JSON inputs")
    export.add_argument("--tv1-run-dir", required=True, help="TV1 run directory containing frames.parquet")
    export.add_argument("--output-dir", required=True, help="directory to receive frames.json and audio.json")
    export.add_argument("--overwrite", action="store_true", help="replace existing output manifests")
    search = subcommands.add_parser("search", help="emit TV4-ready candidates from one WP04 source")
    search.add_argument("--run-dir", required=True)
    search.add_argument("--preprocess-run-id", required=True)
    search.add_argument("--artifact-set-id", required=True)
    search.add_argument("--source", choices=("ocr", "asr", "metadata"), required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--query-id", default="cli-query")
    search.add_argument("--limit", type=int, default=100, choices=range(1, 101))
    validate = subcommands.add_parser("validate", help="write the WP04 promotion-gate report")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--preprocess-run-id", required=True)
    validate.add_argument("--artifact-set-id", required=True)
    validate.add_argument("--frames-json", required=True, help="TV1 FrameRecord JSON array")
    preprocess = subcommands.add_parser("preprocess", help="run config-selected WP04 adapters over TV1 record JSON")
    preprocess.add_argument("--run-dir", required=True)
    preprocess.add_argument("--preprocess-run-id", required=True)
    preprocess.add_argument("--artifact-set-id", required=True)
    preprocess.add_argument("--frames-json", required=True)
    preprocess.add_argument("--audio-json", required=True)
    preprocess.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "doctor":
            return _doctor(arguments)
        if arguments.command == "export-tv1-manifests":
            return _export_tv1_manifests(arguments)
        if arguments.command == "search":
            return _search(arguments)
        if arguments.command == "validate":
            return _validate(arguments)
        if arguments.command == "preprocess":
            return _preprocess(arguments)
    except (OSError, ValueError, KeyError, TV1ManifestExportError) as error:
        print(f"wp04: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
