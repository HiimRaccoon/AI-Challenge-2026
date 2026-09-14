"""Cross-artifact validation before a WP04 artifact set is promoted."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Sequence

from .contracts import ASRSegment, FrameRecord, MetadataRecord, OCRDetection, ObjectDetection, WP04RunIdentity
from .evidence import EvidenceResolver
from .storage import ArtifactStore


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.is_valid,
            "errors": [{"code": issue.code, "message": issue.message} for issue in self.errors],
        }


def validate_records(
    frames: Sequence[FrameRecord], ocr: Sequence[OCRDetection], asr: Sequence[ASRSegment],
    objects: Sequence[ObjectDetection], metadata: Sequence[MetadataRecord],
) -> ValidationReport:
    report = ValidationReport()
    valid_frames = {(frame.video_id, frame.frame_id) for frame in frames}
    for name, records in (("OCR", ocr), ("OBJECT", objects), ("METADATA", metadata)):
        for record in records:
            if (record.video_id, record.frame_id) not in valid_frames:
                report.errors.append(ValidationIssue(f"{name}_FRAME_NOT_IN_TV1", f"{name} references {record.video_id}/{record.frame_id}"))
    try:
        resolver = EvidenceResolver([*ocr, *asr, *objects, *metadata])
    except ValueError as error:
        report.errors.append(ValidationIssue("DUPLICATE_EVIDENCE_REF", str(error)))
        return report
    for record in metadata:
        for reference in record.evidence_refs:
            try:
                resolver.resolve_evidence(reference)
            except KeyError:
                report.errors.append(ValidationIssue("UNKNOWN_EVIDENCE_REF", f"metadata {record.record_id} references {reference}"))
    return report


def validate_run(run_dir: Path, identity: WP04RunIdentity, frames: Sequence[FrameRecord]) -> ValidationReport:
    """Validate persisted shards and write the machine-readable promotion gate."""
    store = ArtifactStore(Path(run_dir), identity)
    ocr = [
        OCRDetection(row["preprocess_run_id"], row["video_id"], row["frame_id"], row["timestamp_ms"],
                     row["raw_text"], row["normalized_text"], tuple(row["bbox_xyxy_norm"]), row["confidence"],
                     row["model_name"], row["model_version"], row["evidence_id"])
        for row in store.read_all_records("ocr")
    ]
    asr = [
        ASRSegment(row["preprocess_run_id"], row["video_id"], row["segment_id"], row["start_ms"], row["end_ms"],
                   row["raw_text"], row["normalized_text"], row["confidence"], row["model_name"], row["model_version"])
        for row in store.read_all_records("asr")
    ]
    objects = [
        ObjectDetection(row["preprocess_run_id"], row["video_id"], row["frame_id"], row["timestamp_ms"],
                        row["label"], tuple(row["bbox_xyxy_norm"]), row["confidence"], row["model_name"],
                        row["model_version"], row["evidence_id"])
        for row in store.read_all_records("object")
    ]
    metadata = [
        MetadataRecord(row["preprocess_run_id"], row["video_id"], row["frame_id"], row["timestamp_ms"],
                       row["fields"], tuple(row["evidence_refs"]), row["record_id"])
        for row in store.read_all_records("metadata")
    ]
    report = validate_records(frames, ocr, asr, objects, metadata)
    report_path = store.root / "reports" / "wp04-validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report
