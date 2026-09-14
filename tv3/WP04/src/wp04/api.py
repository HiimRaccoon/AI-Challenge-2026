"""HTTP wrapper for TV3/WP04 (AIC26-main/TV3/WP04) artifacts.

Exposes the same route/contract that tv4/src/tv4/clients/tv3_client.py
already expects (POST /text/search, /ocr/search, /asr/search,
/object/search, /metadata/search with SearchRequest -> list[SearchCandidate]
JSON), so TV4's original HTTP client works with ZERO code changes on the
TV4 side. This is the "no TV4 edits" alternative to Wp04LocalClient.

Run:
    pip install fastapi uvicorn   # add to TV3/WP04's own venv
    $env:WP04_API_RUN_DIR = "data\\runs\\run-a"
    $env:WP04_API_PREPROCESS_RUN_ID = "run_v1_batch1"
    $env:WP04_API_ARTIFACT_SET_ID = "wp04-a"
    python -m uvicorn wp04.api:app --host 127.0.0.1 --port 8100

Place this file at TV3/WP04/src/wp04/api.py.
"""
from __future__ import annotations

import os
import json
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .contracts import WP04RunIdentity
from .object_enrichment import build_alias_map, canonical_label
from .retrieval import LocalTextIndex
from .retrieval import TextHit
from .semantic_transcript import semantic_search
from .normalization import normalize_text
from .storage import ArtifactStore

_REFERENCE_KEY = {"ocr": "evidence_id", "asr": "segment_id", "metadata": "record_id"}


class SearchRequestIn(BaseModel):
    schema_version: str | None = None
    query_id: str
    task: str
    query_text: str | None = None
    question: str | None = None
    events: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = 100
    language: str | None = "vi"
    session_id: str | None = None
    event_index: int | None = None


class EvidenceWindowRequest(BaseModel):
    """Read nearby evidence for a known TV1 frame without re-running search."""

    video_id: str
    timestamp_ms: int = Field(ge=0)
    window_ms: int = Field(default=3_000, ge=0, le=30_000)
    limit: int = Field(default=50, ge=1, le=100)


class TranscriptWindowSearchRequest(BaseModel):
    """Search spoken transcript and return a bounded, frame-addressable window.

    ``/asr/search`` remains the compatibility endpoint that returns one
    ``SearchCandidate`` per ASR segment. This request is deliberately a
    separate contract because downstream callers need the whole spoken
    interval (and all of its canonical frames) to scope later visual/OCR
    retrieval, not just a midpoint anchor.
    """

    query_id: str
    query_text: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=100)
    context_ms: int = Field(default=10_000, ge=0, le=30_000)
    max_frame_ids: int = Field(default=200, ge=1, le=500)
    mode: Literal["lexical", "semantic", "hybrid"] = "hybrid"


def _text_for_row(modality: str, row: dict[str, Any]) -> str:
    if modality in ("ocr", "asr"):
        return str(row.get("raw_text", ""))
    return str(row.get("fields", ""))


# def _get_store() -> ArtifactStore:
#     run_dir = os.environ.get("WP04_API_RUN_DIR")
#     preprocess_run_id = os.environ.get("WP04_API_PREPROCESS_RUN_ID")
#     artifact_set_id = os.environ.get("WP04_API_ARTIFACT_SET_ID")
#     if not (run_dir and preprocess_run_id and artifact_set_id):
#         raise HTTPException(
#             status_code=500,
#             detail="WP04_API_RUN_DIR / WP04_API_PREPROCESS_RUN_ID / WP04_API_ARTIFACT_SET_ID not set",
#         )
#     identity = WP04RunIdentity(preprocess_run_id, artifact_set_id, "api", "api")
#     return ArtifactStore(Path(run_dir), identity)


def _get_store() -> ArtifactStore:
    run_dir = os.environ.get("WP04_API_RUN_DIR") or str(
        Path(__file__).parents[2] / "data" / "runs" / "run_v3_ultradense_shot_aware"
    )
    preprocess_run_id = os.environ.get("WP04_API_PREPROCESS_RUN_ID") or "run_v3_ultradense"
    artifact_set_id = os.environ.get("WP04_API_ARTIFACT_SET_ID") or "multimodal-v3-shot-aware"
    identity = WP04RunIdentity(preprocess_run_id, artifact_set_id, "api", "api")
    return ArtifactStore(Path(run_dir), identity)




# Both maps are keyed on (store.root, modality) so results stay correct if
# the process is ever pointed at more than one run directory.
_TEXT_INDEX_CACHE: dict[tuple[str, str], tuple[LocalTextIndex, dict[str, dict[str, Any]]]] = {}
_OBJECT_ROWS_CACHE: dict[str, list[dict[str, Any]]] = {}
_FRAME_INDEX_CACHE: dict[str, dict[str, tuple[list[int], list[tuple[int, int]]]]] = {}
_ASR_ROWS_BY_VIDEO_CACHE: dict[str, dict[str, list[dict[str, Any]]]] = {}


def _get_text_index(store: ArtifactStore, modality: str) -> tuple[LocalTextIndex, dict[str, dict[str, Any]]]:
    """Build (and cache) the BM25 index for a modality once per server process.

    Rebuilding LocalTextIndex from every Parquet shard on disk on every single
    search call is fine for one-off KIS/VQA queries (3 TV3 calls) but multiplies
    badly for TRAKE, which fans out to ~3 TV3 calls per event on top of the
    whole-query pass -- repeated full-corpus reads + BM25 rebuilds were the
    dominant cost pushing individual calls past the client's timeout.
    """
    cache_key = (str(store.root), modality)
    cached = _TEXT_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows = store.read_all_records(modality)
    if not rows:
        # No artifact built yet for this modality -> "no candidates", not an error.
        raise HTTPException(status_code=503, detail=f"no {modality} artifacts for this run")

    ref_key = _REFERENCE_KEY[modality]
    documents = [(str(row[ref_key]), _text_for_row(modality, row)) for row in rows]
    by_reference = {str(row[ref_key]): row for row in rows}
    index = LocalTextIndex.from_documents(documents)
    _TEXT_INDEX_CACHE[cache_key] = (index, by_reference)
    return index, by_reference


def _frames_json_path(store: ArtifactStore) -> Path | None:
    """Find TV1's canonical frames export for the active WP04 run.

    The API does not calculate a frame ID from FPS. It only projects an ASR
    time interval onto an explicit canonical TV1 frame record. Deployments
    may set ``WP04_API_FRAMES_JSON``; the standard workspace layout is
    discovered as a convenience for the existing local run.
    """
    configured = os.environ.get("WP04_API_FRAMES_JSON")
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    candidate = store.run_dir.parents[1] / "inputs" / store.identity.preprocess_run_id / "frames.json"
    return candidate if candidate.is_file() else None


def _get_frame_index(store: ArtifactStore) -> dict[str, tuple[list[int], list[tuple[int, int]]]]:
    """Cache canonical frames as (timestamps, (timestamp, frame_id)) per video."""
    cache_key = str(store.root)
    cached = _FRAME_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    path = _frames_json_path(store)
    if path is None:
        _FRAME_INDEX_CACHE[cache_key] = {}
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _FRAME_INDEX_CACHE[cache_key] = {}
        return {}
    grouped: dict[str, list[tuple[int, int]]] = {}
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            video_id, timestamp_ms, frame_id = row.get("video_id"), row.get("timestamp_ms"), row.get("frame_id")
            if (
                isinstance(video_id, str)
                and isinstance(timestamp_ms, int) and not isinstance(timestamp_ms, bool) and timestamp_ms >= 0
                and isinstance(frame_id, int) and not isinstance(frame_id, bool) and frame_id >= 0
            ):
                grouped.setdefault(video_id, []).append((timestamp_ms, frame_id))
    index: dict[str, tuple[list[int], list[tuple[int, int]]]] = {}
    for video_id, frames in grouped.items():
        ordered = sorted(frames, key=lambda item: (item[0], item[1]))
        index[video_id] = ([item[0] for item in ordered], ordered)
    _FRAME_INDEX_CACHE[cache_key] = index
    return index


def _asr_canonical_anchor(row: dict[str, Any], frame_index: dict[str, tuple[list[int], list[tuple[int, int]]]]) -> tuple[int, int] | None:
    """Resolve an ASR segment midpoint to one explicit TV1 canonical frame."""
    video_id = row.get("video_id")
    start_ms, end_ms = row.get("start_ms"), row.get("end_ms")
    if not isinstance(video_id, str) or not isinstance(start_ms, int) or not isinstance(end_ms, int):
        return None
    if isinstance(start_ms, bool) or isinstance(end_ms, bool) or start_ms < 0 or end_ms < start_ms:
        return None
    timestamps, frames = frame_index.get(video_id, ([], []))
    if not frames:
        return None
    midpoint = (start_ms + end_ms) // 2
    position = bisect_left(timestamps, midpoint)
    choices = frames[max(0, position - 1): position + 1]
    if not choices:
        return None
    timestamp_ms, frame_id = min(choices, key=lambda item: (abs(item[0] - midpoint), item[0], item[1]))
    return frame_id, timestamp_ms


def _canonical_frames_in_range(
    video_id: str,
    start_ms: int,
    end_ms: int,
    frame_index: dict[str, tuple[list[int], list[tuple[int, int]]]],
    *,
    limit: int,
) -> list[dict[str, int]]:
    """Return explicit TV1 canonical frames inside an ASR time interval.

    This never estimates a frame via FPS. Returning both IDs and timestamps
    lets TV4 pass an auditable candidate scope to a visual reranker later.
    """
    timestamps, frames = frame_index.get(video_id, ([], []))
    if not frames:
        return []
    left = bisect_left(timestamps, start_ms)
    right = bisect_right(timestamps, end_ms)
    scoped = frames[left:right]
    if len(scoped) > limit:
        # Keep coverage across the entire spoken range rather than returning
        # only early frames. Every returned item is still a canonical frame.
        step = (len(scoped) - 1) / (limit - 1) if limit > 1 else 0
        scoped = [scoped[round(index * step)] for index in range(limit)]
    return [{"frame_id": frame_id, "timestamp_ms": timestamp_ms} for timestamp_ms, frame_id in scoped]


def _get_asr_rows_by_video(store: ArtifactStore) -> dict[str, list[dict[str, Any]]]:
    """Cache ASR rows ordered by time without re-reading the artifact set."""
    cache_key = str(store.root)
    cached = _ASR_ROWS_BY_VIDEO_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _, by_reference = _get_text_index(store, "asr")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in by_reference.values():
        video_id = row.get("video_id")
        start_ms, end_ms = row.get("start_ms"), row.get("end_ms")
        if (
            isinstance(video_id, str)
            and isinstance(start_ms, int) and not isinstance(start_ms, bool)
            and isinstance(end_ms, int) and not isinstance(end_ms, bool)
            and start_ms >= 0 and end_ms >= start_ms
        ):
            grouped.setdefault(video_id, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["start_ms"]), int(row["end_ms"]), str(row.get("segment_id", ""))))
    _ASR_ROWS_BY_VIDEO_CACHE[cache_key] = grouped
    return grouped


def _transcript_windows(
    store: ArtifactStore, request: TranscriptWindowSearchRequest,
) -> list[dict[str, Any]]:
    """Expand ranked ASR hits into temporal transcript windows.

    The lexical index is shared with ``/asr/search``. Thus the first request
    builds it once and later transcript searches do not scan Parquet again.
    A window is assembled only around an actual hit, avoiding a large derived
    corpus of mostly silent fixed-duration windows.
    """
    index, by_reference = _get_text_index(store, "asr")
    rows_by_video = _get_asr_rows_by_video(store)
    frame_index = _get_frame_index(store)
    # BM25 scans the full transcript corpus. In semantic mode it previously
    # ran needlessly before the vector result and added ~22 seconds.
    lexical_hits = index.search(request.query_text, request.limit) if request.mode != "semantic" else []
    semantic_hits = semantic_search(request.query_text, request.limit) if request.mode != "lexical" else None
    if request.mode == "semantic" and semantic_hits is not None:
        hits = [hit for hit in semantic_hits if hit.document_id in by_reference]
    elif semantic_hits:
        # Rank fusion is robust to score-scale differences between BM25 and
        # cosine similarity and keeps lexical matches when the API is noisy.
        fused: dict[str, float] = {}
        for rank, hit in enumerate(lexical_hits, start=1):
            fused[hit.document_id] = fused.get(hit.document_id, 0.0) + 1.0 / (60 + rank)
        for rank, hit in enumerate(semantic_hits, start=1):
            if hit.document_id in by_reference:
                fused[hit.document_id] = fused.get(hit.document_id, 0.0) + 1.0 / (60 + rank)
        hits = [TextHit(document_id, score) for document_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:request.limit]]
    else:
        # Semantic-only also degrades to lexical by contract; an external API
        # outage must not turn spoken retrieval into an empty branch.
        if request.mode == "semantic":
            lexical_hits = index.search(request.query_text, request.limit)
        hits = lexical_hits
    windows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for rank, hit in enumerate(hits, start=1):
        matched = by_reference[hit.document_id]
        video_id = matched.get("video_id")
        start_ms, end_ms = matched.get("start_ms"), matched.get("end_ms")
        if not isinstance(video_id, str) or not isinstance(start_ms, int) or not isinstance(end_ms, int):
            continue
        lower, upper = max(0, start_ms - request.context_ms), end_ms + request.context_ms
        nearby = [
            row for row in rows_by_video.get(video_id, [])
            if int(row["start_ms"]) <= upper and int(row["end_ms"]) >= lower
        ]
        if not nearby:
            continue
        window_start = min(int(row["start_ms"]) for row in nearby)
        window_end = max(int(row["end_ms"]) for row in nearby)
        key = (video_id, window_start, window_end)
        if key in seen:
            continue
        seen.add(key)
        canonical_frames = _canonical_frames_in_range(
            video_id, window_start, window_end, frame_index, limit=request.max_frame_ids,
        )
        windows.append({
            "query_id": request.query_id,
            "video_id": video_id,
            "start_ms": window_start,
            "end_ms": window_end,
            "anchor_segment_id": str(matched.get("segment_id", "")),
            "rank": rank,
            "raw_score": hit.score,
            "confidence": matched.get("confidence"),
            "transcript": " ".join(str(row.get("raw_text", "")).strip() for row in nearby if str(row.get("raw_text", "")).strip()),
            "evidence_refs": [str(row.get("segment_id", "")) for row in nearby if row.get("segment_id")],
            "canonical_frames": canonical_frames,
            "preprocess_run_id": matched.get("preprocess_run_id", "unknown"),
        })
    return windows


def _search_text_modality(store: ArtifactStore, modality: str, request: SearchRequestIn) -> list[dict[str, Any]]:
    ref_key = _REFERENCE_KEY[modality]
    index, by_reference = _get_text_index(store, modality)
    hits = index.search(request.query_text or "", request.limit)

    candidates: list[dict[str, Any]] = []
    frame_index = _get_frame_index(store) if modality == "asr" else {}
    for rank, hit in enumerate(hits, start=1):
        row = by_reference[hit.document_id]
        frame_id, timestamp_ms = row.get("frame_id"), row.get("timestamp_ms")
        if modality == "asr" and (frame_id is None or timestamp_ms is None):
            anchor = _asr_canonical_anchor(row, frame_index)
            if anchor is None:
                continue
            frame_id, timestamp_ms = anchor
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
            continue
        candidates.append({
            "query_id": request.query_id,
            "video_id": row["video_id"],
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "source": modality,
            "rank": rank,
            "preprocess_run_id": row.get("preprocess_run_id", "unknown"),
            "raw_score": hit.score,
            "evidence_refs": [str(row[ref_key])],
            "confidence": row.get("confidence"),
        })
    return candidates


def _get_object_rows(store: ArtifactStore) -> list[dict[str, Any]]:
    cache_key = str(store.root)
    cached = _OBJECT_ROWS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rows = store.read_all_records("object")
    if not rows:
        raise HTTPException(status_code=503, detail="no object artifacts for this run")
    _OBJECT_ROWS_CACHE[cache_key] = rows
    return rows


def _requested_object_labels(query_text: str, alias_map: dict[str, str]) -> set[str]:
    folded = f" {normalize_text(query_text).without_diacritics} "
    requested: set[str] = set()
    for alias, canonical in alias_map.items():
        alias_folded = normalize_text(alias).without_diacritics
        if alias_folded and f" {alias_folded} " in folded:
            requested.add(canonical)
    return requested


def _search_object(store: ArtifactStore, request: SearchRequestIn) -> list[dict[str, Any]]:
    rows = _get_object_rows(store)

    alias_map = build_alias_map()
    requested_labels = _requested_object_labels(request.query_text or "", alias_map)
    if not requested_labels:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        canon, _ = canonical_label(str(row.get("label", "")), alias_map)
        if canon in requested_labels:
            scored.append((float(row.get("confidence") or 0.0), row))
    scored.sort(key=lambda item: item[0], reverse=True)

    candidates: list[dict[str, Any]] = []
    for rank, (score, row) in enumerate(scored[: request.limit], start=1):
        candidates.append({
            "query_id": request.query_id,
            "video_id": row["video_id"],
            "frame_id": int(row["frame_id"]),
            "timestamp_ms": int(row["timestamp_ms"]),
            "source": "object",
            "rank": rank,
            "preprocess_run_id": row.get("preprocess_run_id", "unknown"),
            "raw_score": score,
            "evidence_refs": [str(row.get("evidence_id", ""))],
            "confidence": row.get("confidence"),
        })
    return candidates


app = FastAPI(title="wp04-api", description="HTTP shim over TV3/WP04 artifacts for TV4's TV3Client")


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _get_store()
    except HTTPException:
        return {"status": "degraded"}
    return {"status": "ok"}


def _dedupe_texts(rows: list[dict[str, Any]], *, value_key: str, timestamp_ms: int, limit: int) -> list[str]:
    """Return nearest non-empty evidence values, keeping first occurrence."""
    ordered = sorted(
        rows,
        key=lambda row: (
            abs(int(row.get("timestamp_ms", timestamp_ms)) - timestamp_ms),
            -float(row.get("confidence") or 0.0),
        ),
    )
    values: list[str] = []
    seen: set[str] = set()
    for row in ordered:
        value = str(row.get(value_key, "")).strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
        if len(values) >= limit:
            break
    return values


@app.post("/evidence/window")
def evidence_window(request: EvidenceWindowRequest) -> dict[str, Any]:
    """Return OCR/ASR/object evidence around a candidate's canonical time.

    This reads only the requested video's Parquet shards. A VQA answerer
    therefore gets local evidence even when its words differ from the search
    query that originally retrieved the candidate.
    """
    store = _get_store()
    lower, upper = request.timestamp_ms - request.window_ms, request.timestamp_ms + request.window_ms
    ocr_rows = [
        row for row in store.read_records("ocr", request.video_id)
        if abs(int(row.get("timestamp_ms", -1)) - request.timestamp_ms) <= request.window_ms
    ]
    object_rows = [
        row for row in store.read_records("object", request.video_id)
        if abs(int(row.get("timestamp_ms", -1)) - request.timestamp_ms) <= request.window_ms
    ]
    asr_rows = [
        row for row in store.read_records("asr", request.video_id)
        if int(row.get("start_ms", -1)) <= upper and int(row.get("end_ms", -1)) >= lower
    ]
    asr_for_sort = [
        {
            **row,
            "timestamp_ms": (int(row.get("start_ms", 0)) + int(row.get("end_ms", 0))) // 2,
        }
        for row in asr_rows
    ]
    return {
        "ocr_texts": _dedupe_texts(ocr_rows, value_key="raw_text", timestamp_ms=request.timestamp_ms, limit=request.limit),
        "asr_texts": _dedupe_texts(asr_for_sort, value_key="raw_text", timestamp_ms=request.timestamp_ms, limit=request.limit),
        "object_labels": _dedupe_texts(object_rows, value_key="label", timestamp_ms=request.timestamp_ms, limit=request.limit),
        # Keep original frame identity beside bounded supplementary OCR text.
        # A downstream VQA proposal may then cite the frame that actually
        # displays the answer instead of its earlier ASR retrieval anchor.
        "ocr_records": [
            {
                "detection_id": row.get("evidence_id"), "video_id": row["video_id"],
                "frame_id": int(row["frame_id"]), "timestamp_ms": int(row["timestamp_ms"]),
                "raw_text": row.get("raw_text", ""), "normalized_text": row.get("normalized_text", ""),
                "bbox_xyxy_norm": row.get("bbox_xyxy_norm"), "confidence": row.get("confidence"),
                "model_name": row.get("model_name"), "model_version": row.get("model_version"),
                "preprocess_run_id": row.get("preprocess_run_id"),
            }
            for row in ocr_rows[:request.limit]
        ],
    }


# TV4's rich VQA evidence pack (WP11) prefers these producer-detail catalogue
# endpoints over the bare search-candidate text when available, for bounding
# boxes / ASR word timing / provenance. They degrade gracefully on the TV4
# side (404 -> that evidence branch marked "unavailable", not fatal) so
# adding them is additive and safe for already-running TV4 instances.
@app.get("/ocr/detections")
def ocr_detections(video_id: str, frame_id: int, limit: int = 100) -> list[dict[str, Any]]:
    rows = [r for r in _get_store().read_records("ocr", video_id) if int(r.get("frame_id", -1)) == frame_id]
    return [
        {
            "detection_id": row.get("evidence_id"), "video_id": row["video_id"],
            "frame_id": int(row["frame_id"]), "timestamp_ms": int(row["timestamp_ms"]),
            "raw_text": row.get("raw_text", ""), "normalized_text": row.get("normalized_text", ""),
            "bbox_xyxy_norm": row.get("bbox_xyxy_norm"), "confidence": row.get("confidence"),
            "model_name": row.get("model_name"), "model_version": row.get("model_version"),
            "preprocess_run_id": row.get("preprocess_run_id"),
        }
        for row in rows[:limit]
    ]


@app.get("/asr/segments")
def asr_segments_detail(video_id: str, start_ms: int, end_ms: int, limit: int = 100) -> list[dict[str, Any]]:
    rows = [
        r for r in _get_store().read_records("asr", video_id)
        if int(r.get("start_ms", -1)) <= end_ms and int(r.get("end_ms", -1)) >= start_ms
    ]
    return [
        {
            "segment_id": row.get("segment_id"), "video_id": row["video_id"],
            "start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"]),
            "text": row.get("raw_text", ""), "normalized_text": row.get("normalized_text", ""),
            "confidence": row.get("confidence"), "model_name": row.get("model_name"),
            "model_version": row.get("model_version"), "preprocess_run_id": row.get("preprocess_run_id"),
        }
        for row in rows[:limit]
    ]


@app.get("/asr/{segment_id}/context")
def asr_context(segment_id: str, window: int = 2) -> dict[str, list[dict[str, Any]]]:
    """Neighboring ASR segments (same video, ordered by time) around one segment."""
    try:
        _, by_reference = _get_text_index(_get_store(), "asr")
    except HTTPException:
        return {"segments": []}
    target = by_reference.get(segment_id)
    if target is None:
        return {"segments": []}
    same_video = sorted(
        (row for row in by_reference.values() if row["video_id"] == target["video_id"]),
        key=lambda row: int(row["start_ms"]),
    )
    index = next(i for i, row in enumerate(same_video) if row.get("segment_id") == segment_id)
    lo, hi = max(0, index - window), min(len(same_video), index + window + 1)
    return {
        "segments": [
            {
                "segment_id": row.get("segment_id"), "video_id": row["video_id"],
                "start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"]),
                "text": row.get("raw_text", ""), "normalized_text": row.get("normalized_text", ""),
            }
            for row in same_video[lo:hi]
        ]
    }


@app.get("/objects/detections")
def object_detections(video_id: str, frame_id: int, limit: int = 100) -> list[dict[str, Any]]:
    rows = [r for r in _get_store().read_records("object", video_id) if int(r.get("frame_id", -1)) == frame_id]
    return [
        {
            "detection_id": row.get("evidence_id"), "video_id": row["video_id"],
            "frame_id": int(row["frame_id"]), "timestamp_ms": int(row["timestamp_ms"]),
            "label": row.get("label", ""), "bbox_xyxy_norm": row.get("bbox_xyxy_norm"),
            "confidence": row.get("confidence"), "model_name": row.get("model_name"),
            "model_version": row.get("model_version"), "preprocess_run_id": row.get("preprocess_run_id"),
        }
        for row in rows[:limit]
    ]


@app.get("/metadata/records")
def metadata_records_detail(video_id: str, limit: int = 100) -> list[dict[str, Any]]:
    _METADATA_VALUE_KEYS = (
        "title", "description", "tags", "channel", "upload_date", "language",
        "text", "normalized_text", "raw_fields", "collection_id", "collection_label",
    )
    rows = _get_store().read_records("metadata", video_id)
    results: list[dict[str, Any]] = []
    for row in rows[:limit]:
        fields = row.get("fields") or {}
        record = {
            "metadata_id": row.get("record_id", ""), "video_id": row["video_id"], "source": "wp04_metadata",
            "window_start_ms": None, "window_end_ms": None, "preprocess_run_id": row.get("preprocess_run_id"),
        }
        record.update({key: fields[key] for key in _METADATA_VALUE_KEYS if key in fields})
        results.append(record)
    return results


@app.post("/ocr/search")
def ocr_search(request: SearchRequestIn) -> list[dict[str, Any]]:
    return _search_text_modality(_get_store(), "ocr", request)


@app.post("/asr/search")
def asr_search(request: SearchRequestIn) -> list[dict[str, Any]]:
    return _search_text_modality(_get_store(), "asr", request)


@app.post("/asr/windows/search")
def asr_windows_search(request: TranscriptWindowSearchRequest) -> dict[str, Any]:
    """Find spoken transcript matches and expose an exact temporal frame scope.

    Consumers should use ``canonical_frames`` as a candidate filter for
    visual/OCR/object reranking. Empty ``canonical_frames`` is explicit:
    callers must not synthesize IDs from FPS when the TV1 mapping is absent.
    """
    return {
        "query_id": request.query_id,
        "windows": _transcript_windows(_get_store(), request),
    }


@app.post("/metadata/search")
def metadata_search(request: SearchRequestIn) -> list[dict[str, Any]]:
    return _search_text_modality(_get_store(), "metadata", request)


@app.post("/object/search")
def object_search(request: SearchRequestIn) -> list[dict[str, Any]]:
    return _search_object(_get_store(), request)


@app.post("/text/search")
def text_search(request: SearchRequestIn) -> list[dict[str, Any]]:
    # WP04 has no dedicated "text" modality; merge ocr + asr like Wp04LocalClient did.
    store = _get_store()
    combined: list[dict[str, Any]] = []
    for modality in ("ocr", "asr"):
        try:
            combined.extend(_search_text_modality(store, modality, request))
        except HTTPException:
            continue
    combined.sort(key=lambda c: c["raw_score"] or 0.0, reverse=True)
    return combined[: request.limit]
