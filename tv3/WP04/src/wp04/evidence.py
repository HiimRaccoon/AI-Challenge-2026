"""Stable modality evidence identifiers and a lookup seam for validation/retrieval."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeAlias

from .contracts import ASRSegment, MetadataRecord, OCRDetection, ObjectDetection


EvidenceRecord: TypeAlias = OCRDetection | ASRSegment | ObjectDetection | MetadataRecord


def assign_ocr_evidence_ids(records: list[OCRDetection]) -> list[OCRDetection]:
    """Assign deterministic IDs without changing the caller's record order."""
    ordered = sorted(
        enumerate(records),
        key=lambda item: (
            item[1].video_id, item[1].frame_id, item[1].bbox_xyxy_norm,
            item[1].normalized_text, item[1].raw_text,
        ),
    )
    replacements = {
        index: replace(record, evidence_id=f"ocr:{record.video_id}:{record.frame_id}:{rank}")
        for rank, (index, record) in enumerate(ordered)
    }
    return [replacements[index] for index in range(len(records))]


class EvidenceResolver:
    def __init__(self, records: list[EvidenceRecord]) -> None:
        lookup: dict[str, EvidenceRecord] = {}
        for record in records:
            reference = self._reference_for(record)
            if reference in lookup:
                raise ValueError(f"duplicate evidence reference: {reference}")
            lookup[reference] = record
        self._lookup = lookup

    @staticmethod
    def _reference_for(record: EvidenceRecord) -> str:
        if isinstance(record, ASRSegment):
            return record.segment_id
        if isinstance(record, MetadataRecord):
            return record.record_id
        return record.evidence_id

    def resolve_evidence(self, reference: str) -> EvidenceRecord:
        try:
            return self._lookup[reference]
        except KeyError as error:
            raise KeyError(f"unknown evidence reference: {reference}") from error
