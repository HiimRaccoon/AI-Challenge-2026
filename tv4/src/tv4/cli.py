"""TV4 CLI.

    python -m tv4 kis   --config configs/default.yaml --query "..." [--out outputs/kis.csv]
    python -m tv4 qa    --config configs/default.yaml --query "..." --question "..." [--out outputs/qa.csv]
    python -m tv4 trake --config configs/default.yaml --query "..." [--out outputs/trake.json]
    python -m tv4 batch --config configs/default.yaml --queries queries.json --out outputs/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .clients.tv1_client import TV1Client
from .clients.tv2_refine_client import TV2RefineClient
from .clients.tv2_visual_client import TV2VisualClient
from .clients.tv3_client import TV3Client
from .qa_dense_reader import DenseOcrReader
from .kis_pipeline import KisServices, run_kis_query
from .media_identity import MediaRecord
from .submission import write_kis_csv, write_qa_csv, write_trake_json
from .submission_exporter import (
    AuthoritativeFrameResolver,
    PredictionValidationError,
    query_specs_from_records,
    validate_resolved_predictions,
    write_predictions_final,
)
from .trake_pipeline import run_trake_query
from .wp11_vqa import RuleBasedFallbackEngine, answer_query


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_services(config: dict) -> KisServices:
    base = Path(config.get("_config_dir", "."))
    tv2v = config["tv2_visual"]
    tv2r = config.get("tv2_refine", {"enabled": False})
    dense = config.get("dense_qa", {})
    visual = TV2VisualClient(
        python_executable=str((base / tv2v["python_executable"]).resolve()),
        wp03_cwd=(base / tv2v["wp03_cwd"]).resolve(),
        artifact_root=(base / tv2v["artifact_root"]).resolve(),
        runtime_root=(base / tv2v["runtime_root"]).resolve() if tv2v.get("runtime_root") else None,
        runtime_profile=(base / tv2v["runtime_profile"]).resolve() if tv2v.get("runtime_profile") else None,
        top_k=tv2v.get("top_k", 100),
          candidate_k_per_model=tv2v.get("candidate_k_per_model", 200),
          enabled=tv2v.get("enabled", True),
          persistent=tv2v.get("persistent", False),
          timeout_seconds=float(tv2v.get("timeout_seconds", 20.0)),
          warmup_enabled=tv2v.get("warmup_enabled", False),
          warmup_timeout_seconds=float(tv2v.get("warmup_timeout_seconds", 180.0)),
          warmup_query=str(tv2v.get("warmup_query", "a person in a lecture slide")),
          warmup_models=tuple(str(value) for value in tv2v.get("warmup_models", ())),
          models=tuple(str(value) for value in tv2v.get("online_models", ())),
          temporal_models=tuple(str(value) for value in tv2v.get("temporal_online_models", ())),
          multi_moment_models=tuple(str(value) for value in tv2v.get("multi_moment_online_models", ())),
      )
    refine = None
    if tv2r.get("enabled"):
        refine = TV2RefineClient(
            python_executable=str((base / tv2r["python_executable"]).resolve()),
            wp09_cwd=(base / tv2r["wp09_cwd"]).resolve(),
            config_path=(base / tv2r["config_path"]).resolve(),
            tv1_base_url=config["tv1"]["base_url"],
            exact_certification_path=_exact_certification_path(base, tv2r),
        )
    media_registry = None
    registry_path = config.get("tv1", {}).get("media_registry_path")
    if registry_path:
        raw_registry = json.loads((base / registry_path).read_text(encoding="utf-8"))
        if not isinstance(raw_registry, list):
            raise ValueError("tv1.media_registry_path must contain a JSON list")
        records = [_media_record_from_authority(item, config.get("preprocess_run_id", "unknown")) for item in raw_registry]
        media_registry = {record.video_id: record for record in records}
        if len(media_registry) != len(records):
            raise ValueError("media registry contains duplicate video_id")
    feedback = None
    tv2f = config.get("tv2_feedback", {})
    if tv2f.get("enabled"):
        from .adapters.wp08_adapter import Wp08FeedbackAdapter
        db_path = (base / tv2f.get("sqlite_path", "sessions.db")).resolve()
        feedback = Wp08FeedbackAdapter(
            db_path=db_path,
            preprocess_run_id=config.get("preprocess_run_id", "unknown"),
            fixture_mode=False,
            pool_provider=lambda q: _build_live_pool_from_visual(visual, q, config.get("preprocess_run_id", "unknown")),
        )
    return KisServices(
        tv1=TV1Client(config["tv1"]["base_url"]),
        tv3=TV3Client(config["tv3"]["base_url"]),
        visual=visual,
        refine=refine,
        feedback=feedback,
        dense_ocr=DenseOcrReader(
            enabled=dense.get("enabled", True),
            base_url=dense.get("ocr_base_url", "http://127.0.0.1:8765"),
            timeout_seconds=float(dense.get("timeout_seconds", 120)),
            max_frames=int(dense.get("max_frames", 64)),
            max_batch_frames=int(dense.get("max_batch_frames", 4)),
            min_confidence=float(dense.get("min_confidence", 0.45)),
            min_stable_frames=int(dense.get("min_stable_frames", 2)),
        ),
        preprocess_run_id=config.get("preprocess_run_id", "unknown"),
        original_media_root=(base / config["tv1"]["original_media_root"]).resolve() if config.get("tv1", {}).get("original_media_root") else None,
        media_registry=media_registry,
        allowed_media_extensions=frozenset(str(value).lower() for value in config.get("tv1", {}).get("allowed_media_extensions", [])),
        derivative_asset_root=(base / config["tv1"]["derivative_asset_root"]).resolve() if config.get("tv1", {}).get("derivative_asset_root") else None,
        allowed_image_extensions=frozenset(str(value).lower() for value in config.get("tv1", {}).get("allowed_image_extensions", [])),
    )


def _build_live_pool_from_visual(visual_client: TV2VisualClient, query: str, preprocess_run_id: str):
    from wp08.contracts import CandidateId, CandidateMetadata, SessionPool
    candidates = visual_client.search("live-wp08-pool", query)
    if not candidates:
        raise ValueError("visual search returned zero candidates for feedback session")
    cid_list = tuple(CandidateId(c.video_id, c.frame_id) for c in candidates)
    meta_list = tuple(
        CandidateMetadata(cid, c.timestamp_ms, c.evidence_refs[0].removeprefix("keyframe:") if c.evidence_refs else f"keyframes/{c.video_id}/{c.frame_id}.jpg")
        for cid, c in zip(cid_list, candidates)
    )
    return SessionPool(
        wp03_run_id=preprocess_run_id,
        candidates=cid_list,
        candidate_metadata=meta_list,
        snapshot={"live": True, "query": query, "pool_size": len(cid_list)},
        provenance={"mode": "live", "source": "wp03_visual"},
    )


def _exact_certification_path(base: Path, config: dict) -> Path:
    env_name = config.get("exact_certification_env")
    if env_name:
        value = os.environ.get(str(env_name))
        if not value:
            raise ValueError(f"{env_name} must name the runtime WP09 certification record")
        return Path(value).resolve(strict=True)
    if config.get("exact_certification_path"):
        return (base / config["exact_certification_path"]).resolve(strict=True)
    raise ValueError("tv2_refine exact certification authority is required")


def _media_record_from_authority(item: object, preprocess_run_id: str) -> MediaRecord:
    if not isinstance(item, dict):
        raise ValueError("media registry contains a non-object record")
    # WP00's corpus manifest is the authoritative path/digest authority.  It
    # predates WP01's time-base field, which WP09 independently obtains from
    # TV1's live media authority before it emits a proof.
    video_id = item.get("video_id", "")
    return MediaRecord(
        video_id=str(video_id),
        original_video_path=str(item.get("original_video_path", "")),
        source_sha256=str(item.get("source_sha256", "")),
        preprocess_run_id=str(item.get("preprocess_run_id", preprocess_run_id)),
        time_base=str(item.get("time_base", "")),
        media_record_ref=str(item.get("media_record_ref", f"manifest/{video_id}")),
        mapping_ref=str(item.get("mapping_ref", f"tv1-frames/{video_id}")),
    )


def _cmd_kis(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    config["_config_dir"] = str(Path(args.config).parent)
    services = build_services(config)
    ranked = run_kis_query(args.query, services, top_k=config["fusion"]["top_k"])
    out = Path(args.out) if args.out else Path(config["output_dir"]) / "kis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_kis_csv(out, ranked[0].query_id if ranked else "kis-query", ranked)
    print(f"Wrote {len(ranked)} ranked candidates to {out}")
    return 0


def _cmd_qa(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    config["_config_dir"] = str(Path(args.config).parent)
    services = build_services(config)
    ranked = run_kis_query(args.query, services, top_k=config["fusion"]["top_k"])
    engine = RuleBasedFallbackEngine()
    results = [answer_query(c, args.question, services.tv1, services.tv3, engine) for c in ranked]
    out = Path(args.out) if args.out else Path(config["output_dir"]) / "qa.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_qa_csv(out, ranked[0].query_id if ranked else "qa-query", results)
    print(f"Wrote {len(results)} ranked (video,frame,answer) rows to {out}")
    return 0


def _cmd_trake(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    config["_config_dir"] = str(Path(args.config).parent)
    services = build_services(config)
    hyp = run_trake_query(args.query, services, strategy=args.strategy)
    out = Path(args.out) if args.out else Path(config["output_dir"]) / "trake.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_trake_json(out, hyp.query_id if hyp else "trake-query", hyp)
    print(f"Wrote TRAKE result ({'ok' if hyp else 'no monotonic alignment found'}) to {out}")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    config["_config_dir"] = str(Path(args.config).parent)
    services = build_services(config)
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in queries:
        task = q["task"]
        qid = q.get("query_id", "q")
        if task == "KIS":
            ranked = run_kis_query(q["query_text"], services, query_id=qid, top_k=config["fusion"]["top_k"])
            write_kis_csv(out_dir / f"{qid}.csv", qid, ranked)
        elif task == "VQA":
            ranked = run_kis_query(q["query_text"], services, query_id=qid, top_k=config["fusion"]["top_k"])
            engine = RuleBasedFallbackEngine()
            results = [answer_query(c, q["question"], services.tv1, services.tv3, engine) for c in ranked]
            write_qa_csv(out_dir / f"{qid}.csv", qid, results)
        elif task == "TRAKE":
            hyp = run_trake_query(q["query_text"], services, events=q.get("events"), query_id=qid)
            write_trake_json(out_dir / f"{qid}.json", qid, hyp)
        else:
            print(f"skip unknown task {task!r} for {qid}", file=sys.stderr)
    print(f"Batch complete: {len(queries)} queries -> {out_dir}")
    return 0


def _cmd_handoff(args: argparse.Namespace) -> int:
    """Run the current batch-query format through TV4 and publish final JSON.

    This is intentionally separate from the legacy ``batch`` command: a
    handoff is only successful when every query is present, every result is
    valid, and every exported frame is confirmed by TV1.
    """
    try:
        config = _load_config(args.config)
        config["_config_dir"] = str(Path(args.config).parent)
        services = build_services(config)
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        if not isinstance(queries, list):
            raise PredictionValidationError("--queries must contain a JSON array")
        specs = query_specs_from_records(queries)
        results: dict[str, object] = {}
        for query in queries:
            query_id = query["query_id"]
            task = query["task"]
            if task == "KIS":
                results[query_id] = run_kis_query(
                    query["query_text"], services, query_id=query_id, top_k=config["fusion"]["top_k"]
                )
            elif task == "VQA":
                ranked = run_kis_query(
                    query["query_text"], services, query_id=query_id, top_k=config["fusion"]["top_k"]
                )
                engine = RuleBasedFallbackEngine()
                results[query_id] = [
                    answer_query(candidate, query["question"], services.tv1, services.tv3, engine)
                    for candidate in ranked[: args.top_k_answers]
                ]
            elif task == "TRAKE":
                results[query_id] = run_trake_query(
                    query["query_text"], services, events=query["events"], query_id=query_id
                )
            else:  # query_specs_from_records already rejects this; retain a safe guard here.
                raise PredictionValidationError(f"{query_id}: unsupported task {task!r}")

        summary = write_predictions_final(
            Path(args.out), specs, results, AuthoritativeFrameResolver(services.tv1)
        )
    except Exception as exc:  # fail cleanly for any retrieval/API/resolver error
        print(f"HANDOFF FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        "HANDOFF PASS: "
        f"queries={summary.query_count} kis={summary.kis_count} "
        f"vqa={summary.vqa_count} trake={summary.trake_count} -> {args.out}"
    )
    return 0


def _cmd_validate_handoff(args: argparse.Namespace) -> int:
    """Independently validate the final JSON and its TV1 frame identities."""
    try:
        config = _load_config(args.config)
        config["_config_dir"] = str(Path(args.config).parent)
        services = build_services(config)
        records = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise PredictionValidationError("--queries must contain a JSON array")
        specs = query_specs_from_records(records)
        predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        summary = validate_resolved_predictions(
            predictions, specs, AuthoritativeFrameResolver(services.tv1)
        )
    except Exception as exc:
        print(f"HANDOFF VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2

    print(
        "HANDOFF VALIDATION PASS: "
        f"queries={summary.query_count} kis={summary.kis_count} "
        f"vqa={summary.vqa_count} trake={summary.trake_count}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tv4")
    sub = parser.add_subparsers(dest="command", required=True)

    p_kis = sub.add_parser("kis")
    p_kis.add_argument("--config", required=True)
    p_kis.add_argument("--query", required=True)
    p_kis.add_argument("--out")
    p_kis.set_defaults(func=_cmd_kis)

    p_qa = sub.add_parser("qa")
    p_qa.add_argument("--config", required=True)
    p_qa.add_argument("--query", required=True)
    p_qa.add_argument("--question", required=True)
    p_qa.add_argument("--out")
    p_qa.set_defaults(func=_cmd_qa)

    p_trake = sub.add_parser("trake")
    p_trake.add_argument("--config", required=True)
    p_trake.add_argument("--query", required=True)
    p_trake.add_argument("--strategy", default="dp", choices=["dp", "greedy"])
    p_trake.add_argument("--out")
    p_trake.set_defaults(func=_cmd_trake)

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--config", required=True)
    p_batch.add_argument("--queries", required=True)
    p_batch.add_argument("--out")
    p_batch.set_defaults(func=_cmd_batch)

    p_handoff = sub.add_parser("handoff", help="write validated predictions_final.json from a TV4 batch query JSON")
    p_handoff.add_argument("--config", required=True)
    p_handoff.add_argument("--queries", required=True)
    p_handoff.add_argument("--out", required=True, help="final predictions_final.json path")
    p_handoff.add_argument("--top-k-answers", type=int, default=5, dest="top_k_answers")
    p_handoff.set_defaults(func=_cmd_handoff)

    p_validate_handoff = sub.add_parser(
        "validate-handoff", help="independently validate predictions_final.json against TV1"
    )
    p_validate_handoff.add_argument("--config", required=True)
    p_validate_handoff.add_argument("--queries", required=True)
    p_validate_handoff.add_argument("--predictions", required=True)
    p_validate_handoff.set_defaults(func=_cmd_validate_handoff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
