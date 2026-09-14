"""JSON-only WP09 command line handoff for TV4/TV5 adapters."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .config import RefinementConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wp09")
    sub = parser.add_subparsers(dest="command", required=True)
    refine = sub.add_parser("refine")
    refine.add_argument("--request", required=True)
    refine.add_argument("--config", required=True)
    refine.add_argument("--decoder-factory", help="module:callable returning a canonical mapped decoder")
    refine.add_argument("--scorer-factory", help="optional module:callable returning a FrameScorer")
    neighbors = sub.add_parser("neighbors")
    neighbors.add_argument("--request", required=True)
    neighbors.add_argument("--resolver-factory", required=True, help="module:callable returning an ExactFrameResolver")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if args.command == "neighbors":
            request = exact_neighbor_request_from_dict(payload)
            resolver = _load_factory(args.resolver_factory)(request)
            result = resolver.resolve(request.candidate, request.video_path, request.context, offsets=request.offsets)
            from dataclasses import asdict
            print(json.dumps(asdict(result), ensure_ascii=False, default=str))
            return 0
        request = request_from_dict(payload)
        config = RefinementConfig.load(Path(args.config))
        if not args.decoder_factory:
            raise ValueError("--decoder-factory is required for canonical media/mapping resolution")
        decoder = _load_factory(args.decoder_factory)(request)
        scorer = _load_factory(args.scorer_factory)(request, config) if args.scorer_factory else None
        from .service import ExactFrameRefiner
        print(json.dumps(ExactFrameRefiner(decoder, scorer, config).refine(request).to_dict(), ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


def _load_factory(spec: str):
    module_name, sep, name = spec.partition(":")
    if not sep or not module_name or not name:
        raise ValueError("factory must use module:callable")
    factory = getattr(importlib.import_module(module_name), name)
    if not callable(factory):
        raise ValueError("factory must be callable")
    return factory


def request_from_dict(data: dict[str, Any]):
    from .contracts import CoarseCandidate, DecodeBudget, EvidenceContribution, RefineRequest, RefinementContext, RefinementPolicy, Task
    candidate = data["candidate"]
    context = data["context"]
    budget = data["decode_budget"]
    return RefineRequest(
        candidate=CoarseCandidate(**candidate), video_path=Path(data["video_path"]), task=Task(data["task"]),
        refinement_text=data["refinement_text"], policy=RefinementPolicy(data["policy"]),
        context=RefinementContext(**context), decode_budget=DecodeBudget(**budget),
        evidence=tuple(EvidenceContribution(**entry) for entry in data.get("evidence", [])), event_index=data.get("event_index"),
    )


def exact_neighbor_request_from_dict(data: dict[str, Any]):
    from .contracts import CoarseCandidate, ExactNeighborRequest, RefinementContext

    return ExactNeighborRequest(
        candidate=CoarseCandidate(**data["candidate"]),
        video_path=Path(data["video_path"]),
        context=RefinementContext(**data["context"]),
        offsets=tuple(data.get("offsets", (0,))),
    )
