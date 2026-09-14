"""Read-only, kind-specific WP03/WP04 artifact readiness validation."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib, json
from pathlib import Path
from typing import Any, Literal

Status = Literal["READY", "PARTIAL", "HANDOVER PENDING", "CODE GAP", "INCOMPATIBLE", "ACTUALLY MISSING"]
Modality = Literal["OCR", "ASR", "Object", "Metadata"]

@dataclass(frozen=True)
class ReadinessConfig:
    component: str; kind: Literal["wp03", "wp04"]; root: Path
    expected_videos: tuple[str, ...] = (); known_handover: bool = False; upstream_capability: bool = False
    expected_run_id: str | None = None; manifest_name: str = "manifest.json"; expected_digest: str | None = None
    modality: Modality | None = None; records_name: str = "records.json"; index_name: str | None = None
    @classmethod
    def wp03(cls, root: Path, *, expected_videos: list[str], expected_run_id: str | None = None, manifest_name: str = "manifest.json", expected_digest: str | None = None) -> "ReadinessConfig":
        return cls("WP03 Visual", "wp03", root, tuple(expected_videos), expected_run_id=expected_run_id, manifest_name=manifest_name, expected_digest=expected_digest)

@dataclass(frozen=True)
class ReadinessReport:
    component: str; status: Status; reason: str; diagnostics: tuple[str, ...] = ()
    observed_coverage: dict[str, int] = field(default_factory=dict); expected_coverage: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict); affected_capability: str = ""; fallback: str = ""
    def to_dict(self) -> dict[str, Any]: return asdict(self)

def _report(c: ReadinessConfig, status: Status, reason: str, diagnostics: list[str] | None = None, **kw: Any) -> ReadinessReport:
    return ReadinessReport(c.component, status, reason, tuple(diagnostics or ()), affected_capability=c.component, fallback="Read-only validation only; do not repair, regenerate, or preprocess.", **kw)

def _load(c: ReadinessConfig, name: str) -> tuple[dict[str, Any] | None, bytes | None, ReadinessReport | None]:
    path = c.root / name
    if not path.exists(): return None, None, _report(c, "INCOMPATIBLE", f"artifact root lacks required {name}")
    try:
        raw = path.read_bytes(); data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e: return None, None, _report(c, "INCOMPATIBLE", f"malformed {name}", [str(e)])
    if not isinstance(data, dict): return None, None, _report(c, "INCOMPATIBLE", f"{name} must be an object")
    return data, raw, None

def validate_readiness(c: ReadinessConfig) -> ReadinessReport:
    if not c.root.exists():
        if c.known_handover: return _report(c, "HANDOVER PENDING", "known owner handover has not been accepted")
        if c.upstream_capability: return _report(c, "CODE GAP", "upstream capability exists but adapter/transport is absent")
        return _report(c, "ACTUALLY MISSING", "no verified asset/source/contract or known handover")
    return _wp03(c) if c.kind == "wp03" else _wp04(c)

def _wp03(c: ReadinessConfig) -> ReadinessReport:
    m, raw, bad = _load(c, c.manifest_name)
    if bad: return bad
    assert m is not None and raw is not None
    if c.expected_digest and hashlib.sha256(raw).hexdigest() != c.expected_digest: return _report(c, "INCOMPATIBLE", "authoritative manifest digest mismatch")
    if c.expected_run_id and m.get("preprocess_run_id") != c.expected_run_id: return _report(c, "INCOMPATIBLE", "incompatible preprocess_run_id")
    vectors = m.get("vector_count", m.get("vectors")); selected = m.get("selected_frames", vectors)
    if not isinstance(vectors, int) or not isinstance(selected, int) or vectors != selected: return _report(c, "INCOMPATIBLE", "index/map/vector count inconsistency")
    if not isinstance(m.get("mapping_sha256", m.get("canonical_mapping")), (str, bool)) or not m.get("mapping_sha256", m.get("canonical_mapping")): return _report(c, "INCOMPATIBLE", "canonical mapping compatibility is not proven")
    if not m.get("model_version") or not (m.get("index_sha256") or m.get("index_version")): return _report(c, "INCOMPATIBLE", "model/index/map provenance is incomplete")
    videos = m.get("videos")
    if isinstance(videos, list): observed_videos = len(set(videos))
    elif isinstance(m.get("unique_video_count"), int): observed_videos = m["unique_video_count"]
    else: return _report(c, "INCOMPATIBLE", "unique-video coverage is not proven by inspected artifact metadata")
    obs={"unique_videos":observed_videos,"selected_frames":selected,"vectors":vectors}; exp={"unique_videos":len(c.expected_videos)} if c.expected_videos else {}
    prov={k:m[k] for k in ("preprocess_run_id","wp03_run_id","model_version","index_sha256","mapping_sha256") if k in m}
    if c.expected_videos and observed_videos != len(c.expected_videos): return _report(c,"PARTIAL","coverage is below authoritative registry",observed_coverage=obs,expected_coverage=exp,provenance=prov)
    return _report(c,"READY","WP03 manifest/index/map evidence is compatible",observed_coverage=obs,expected_coverage=exp,provenance=prov)

def _wp04(c: ReadinessConfig) -> ReadinessReport:
    if c.modality is None: return _report(c,"INCOMPATIBLE","WP04 modality must be explicit")
    m, raw, bad = _load(c,c.manifest_name)
    if bad: return bad
    assert m is not None and raw is not None
    if c.expected_digest and hashlib.sha256(raw).hexdigest()!=c.expected_digest: return _report(c,"INCOMPATIBLE","authoritative manifest digest mismatch")
    if c.expected_run_id and m.get("preprocess_run_id")!=c.expected_run_id: return _report(c,"INCOMPATIBLE","incompatible preprocess_run_id")
    records_path=c.root/c.records_name
    if not records_path.exists() or (c.index_name and not (c.root/c.index_name).exists()): return _report(c,"INCOMPATIBLE","required modality records or retrieval index missing")
    try: records=json.loads(records_path.read_text())
    except (OSError,json.JSONDecodeError) as e: return _report(c,"INCOMPATIBLE","malformed modality records",[str(e)])
    if not isinstance(records,list) or not m.get("model_version") or not m.get("preprocess_run_id"): return _report(c,"INCOMPATIBLE","invalid modality records/provenance")
    required={"OCR":("video_id","frame_id","timestamp_ms","text","bbox"),"ASR":("video_id","start_ms","end_ms","transcript","context"),"Object":("video_id","frame_id","timestamp_ms","label","bbox","crop_ref"),"Metadata":("video_id","evidence_value")}[c.modality]
    for i, row in enumerate(records):
        if not isinstance(row,dict) or any(row.get(k) in (None,"") for k in required): return _report(c,"INCOMPATIBLE",f"{c.modality} canonical linkage/evidence failure",[f"record={i}"])
        if c.modality == "ASR" and (not isinstance(row["start_ms"], int) or not isinstance(row["end_ms"], int) or row["start_ms"] > row["end_ms"]): return _report(c,"INCOMPATIBLE","ASR timing linkage is invalid",[f"record={i}"])
    return _report(c,"READY",f"{c.modality} handover evidence is compatible",observed_coverage={"records":len(records)},provenance={"preprocess_run_id":m["preprocess_run_id"],"model_version":m["model_version"]})
