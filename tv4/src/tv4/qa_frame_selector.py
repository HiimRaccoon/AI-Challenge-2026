"""Select answer-supporting frames from TV1-owned VQA evidence only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Sequence

from .contracts import EvidencePack


def _fold(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", _fold(text)) if len(token) > 1}


def _supports(answer: str, text: str) -> bool:
    answer_folded = _fold(answer).strip()
    text_folded = _fold(text)
    if not answer_folded:
        return False
    if answer_folded in text_folded:
        return True
    answer_tokens = _tokens(answer)
    return bool(answer_tokens) and len(answer_tokens & _tokens(text)) / len(answer_tokens) >= 0.6


def _asr_support_intervals(answer: str, evidence: EvidencePack) -> list[tuple[int, int]]:
    intervals = [
        (item.start_ms, item.end_ms)
        for item in evidence.asr_evidence
        if _supports(answer, item.text)
    ]
    for item in evidence.asr_evidence:
        for row in item.context:
            if not isinstance(row, dict) or not _supports(answer, str(row.get("text", ""))):
                continue
            start_ms, end_ms = row.get("start_ms"), row.get("end_ms")
            if (
                isinstance(start_ms, int) and not isinstance(start_ms, bool)
                and isinstance(end_ms, int) and not isinstance(end_ms, bool)
            ):
                intervals.append((start_ms, end_ms))
    return intervals


@dataclass(frozen=True)
class AnswerFrameCandidate:
    frame_id: int
    timestamp_ms: int
    score: float
    support_modalities: Sequence[str]

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 6)
        payload["support_modalities"] = list(self.support_modalities)
        return payload


def answer_support_span(answer: str, evidence: EvidencePack) -> tuple[int | None, int | None]:
    intervals: list[tuple[int, int]] = _asr_support_intervals(answer, evidence)
    intervals.extend(
        (item.timestamp_ms, item.timestamp_ms)
        for item in evidence.ocr_evidence
        if _supports(answer, item.raw_text)
    )
    intervals.extend(
        (item.timestamp_ms, item.timestamp_ms)
        for item in evidence.object_evidence
        if _supports(answer, item.canonical_label or item.label)
    )
    if not intervals:
        return None, None
    return min(start for start, _ in intervals), max(end for _, end in intervals)


def select_answer_frames(
    answer: str,
    evidence: EvidencePack,
    *,
    verified: bool,
    limit: int = 3,
) -> list[AnswerFrameCandidate]:
    """Rescore only canonical frames supplied by TV1; never calculate an ID."""
    if limit < 1:
        raise ValueError("limit must be positive")
    selected = []
    for frame in evidence.selected_frames:
        modalities: list[str] = []
        score = 0.10 if frame.keyframe_path else 0.05
        if frame.frame_id == evidence.frame_id:
            score += 0.20
            modalities.append("retrieval_anchor")

        if any(item.frame_id == frame.frame_id and _supports(answer, item.raw_text) for item in evidence.ocr_evidence):
            score += 0.35
            modalities.append("ocr")
        if any(
            item.frame_id == frame.frame_id and _supports(answer, item.canonical_label or item.label)
            for item in evidence.object_evidence
        ):
            score += 0.30
            modalities.append("object")

        matching_asr = _asr_support_intervals(answer, evidence)
        if matching_asr:
            distance_ms = min(
                0 if start_ms <= frame.timestamp_ms <= end_ms
                else min(abs(frame.timestamp_ms - start_ms), abs(frame.timestamp_ms - end_ms))
                for start_ms, end_ms in matching_asr
            )
            score += 0.30 * max(0.0, 1.0 - distance_ms / 5_000.0)
            modalities.append("asr")
        if verified:
            score += 0.05
            modalities.append("verifier")

        selected.append(AnswerFrameCandidate(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            score=min(1.0, score),
            support_modalities=tuple(dict.fromkeys(modalities)),
        ))

    selected.sort(key=lambda item: (-item.score, item.timestamp_ms, item.frame_id))
    return selected[:limit]
