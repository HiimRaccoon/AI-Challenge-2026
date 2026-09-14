"""Canonical submission prediction models and basket structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskType = Literal["KIS", "VQA", "TRAKE"]


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    query_id: str | None = None
    task_type: TaskType | None = None
    record_count: int = 0
    package_digest_sha256: str | None = None


@dataclass(frozen=True)
class KisPrediction:
    video_id: str
    frame_id: int
    rank: int = 1
    score: float | None = None
    submission_selectable: bool = True

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.video_id or not isinstance(self.video_id, str):
            errs.append("video_id must be a non-empty string")
        elif self.video_id.endswith(".mp4"):
            errs.append("video_id must not include .mp4 extension")
        if not isinstance(self.frame_id, int) or isinstance(self.frame_id, bool) or self.frame_id < 0:
            errs.append("frame_id must be a non-negative integer")
        if not self.submission_selectable:
            errs.append("prediction candidate is not certified/selectable for submission")
        return errs


@dataclass(frozen=True)
class VqaPrediction:
    video_id: str
    frame_id: int
    approved_answer: str
    rank: int = 1
    approved: bool = False
    score: float | None = None
    submission_selectable: bool = True

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.video_id or not isinstance(self.video_id, str):
            errs.append("video_id must be a non-empty string")
        elif self.video_id.endswith(".mp4"):
            errs.append("video_id must not include .mp4 extension")
        if not isinstance(self.frame_id, int) or isinstance(self.frame_id, bool) or self.frame_id < 0:
            errs.append("frame_id must be a non-negative integer")
        if not self.approved:
            errs.append("VQA answer requires mandatory operator approval before submission")
        if not isinstance(self.approved_answer, str) or not self.approved_answer.strip():
            errs.append("approved_answer must be non-empty string")
        elif len(self.approved_answer) > 100:
            errs.append("approved_answer exceeds provisional maximum length of 100 characters")
        if not self.submission_selectable:
            errs.append("prediction candidate is not certified/selectable for submission")
        return errs


@dataclass(frozen=True)
class TrakePrediction:
    video_id: str
    event_frame_ids: tuple[int, ...]
    rank: int = 1
    score: float | None = None
    expected_event_count: int | None = None
    submission_selectable: bool = True

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.video_id or not isinstance(self.video_id, str):
            errs.append("video_id must be a non-empty string")
        elif self.video_id.endswith(".mp4"):
            errs.append("video_id must not include .mp4 extension")
        if not self.event_frame_ids or len(self.event_frame_ids) < 2:
            errs.append("TRAKE requires at least 2 ordered event frames")
        if self.expected_event_count is not None and len(self.event_frame_ids) != self.expected_event_count:
            errs.append(f"TRAKE event frame count ({len(self.event_frame_ids)}) does not match expected ({self.expected_event_count})")
        for idx, fid in enumerate(self.event_frame_ids):
            if not isinstance(fid, int) or isinstance(fid, bool) or fid < 0:
                errs.append(f"event frame #{idx} must be a non-negative integer")
        if not self.submission_selectable:
            errs.append("prediction candidate is not certified/selectable for submission")
        return errs


BasketItem = KisPrediction | VqaPrediction | TrakePrediction


@dataclass
class Basket:
    query_id: str
    task_type: TaskType
    items: list[BasketItem] = field(default_factory=list)
    max_items: int = 100

    def add(self, item: BasketItem) -> bool:
        """Add item up to max 100; reject additions past 100."""
        if len(self.items) >= self.max_items:
            return False
        self.items.append(item)
        return True

    def remove(self, index: int) -> bool:
        if 0 <= index < len(self.items):
            self.items.pop(index)
            return True
        return False

    def reorder(self, from_idx: int, to_idx: int) -> bool:
        if 0 <= from_idx < len(self.items) and 0 <= to_idx < len(self.items):
            item = self.items.pop(from_idx)
            self.items.insert(to_idx, item)
            return True
        return False

    def audit(self) -> ValidationReport:
        all_errors: list[str] = []
        if not self.items:
            all_errors.append("Basket is empty")
        if len(self.items) > self.max_items:
            all_errors.append(f"Basket contains {len(self.items)} items, exceeding limit of {self.max_items}")

        for i, item in enumerate(self.items):
            errs = item.validate()
            for e in errs:
                all_errors.append(f"Item #{i + 1} ({self.task_type}): {e}")

        return ValidationReport(
            is_valid=not all_errors,
            errors=tuple(all_errors),
            query_id=self.query_id,
            task_type=self.task_type,
            record_count=len(self.items),
        )
