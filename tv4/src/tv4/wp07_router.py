"""WP07 — Query Router.

Turns one raw contest query (Textual KIS / Q&A / TRAKE, as defined in
`Thong-tin-vong-So-tuyen-AIC2026`) into one or more `SearchRequest` objects
plus a routing decision (which of visual/ocr/asr/object/metadata to call).

Routing is rule-based on purpose: it is cheap, auditable, and easy to unit
test. It is intentionally *not* a learned router — the plan (WP07 task in
AIC_2026_PLAN.xlsx) explicitly calls for "rule fallback" and only adding
dynamic/learned fusion after ablation shows it wins.
"""
from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field

from .contracts import SearchRequest, Task

# Cheap cues that a query is expected to be found primarily via on-screen
# text / spoken words / detected objects. These only ever *add* branches;
# the visual branch is always queried because visual is the fallback of
# last resort and the cheapest signal to always have.
_OCR_HINTS = re.compile(r"chữ|dòng chữ|biển hiệu|phụ đề|caption|logo|text on screen|banner", re.I)
_ASR_HINTS = re.compile(r"nói|phát biểu|lời thoại|MC|hát|bài hát|nhạc|say[s]?|speech|announce", re.I)
_OBJECT_HINTS = re.compile(
    r"\b(áo|xe|chó|mèo|bàn|ghế|laptop|điện thoại|xe máy|ô tô|người|car|dog|cat|phone)\b", re.I
)

_LECTURE_TEXT_HINTS = re.compile(
    r"\b(bai giang|giang bai|giao vien|slide|lecture|lesson|teacher|"
    r"de thi|cau hoi so|question number|exam question)\b",
    re.I,
)


def _fold_text(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


@dataclass(frozen=True)
class RouteDecision:
    request: SearchRequest
    branches: tuple[str, ...]  # subset of {"visual","ocr","asr","object","metadata"}


def _new_query_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _branches_for_text(text: str, *, include_metadata: bool) -> tuple[str, ...]:
    branches = ["visual"]
    lecture_text = bool(_LECTURE_TEXT_HINTS.search(_fold_text(text)))
    if _OCR_HINTS.search(text) or lecture_text:
        branches.append("ocr")
    if _ASR_HINTS.search(text) or lecture_text:
        branches.append("asr")
    if _OBJECT_HINTS.search(text):
        branches.append("object")
    if include_metadata:
        branches.append("metadata")
    return tuple(dict.fromkeys(branches))  # stable de-dup


def route_kis(query_text: str, *, query_id: str | None = None, limit: int = 100) -> RouteDecision:
    """Textual KIS: locate one video/frame from a full natural-language description."""
    request = SearchRequest(
        query_id=query_id or _new_query_id("kis"),
        task="KIS",
        query_text=query_text,
        limit=limit,
    )
    return RouteDecision(request, _branches_for_text(query_text, include_metadata=True))


def route_qa(description: str, question: str, *, query_id: str | None = None, limit: int = 100) -> RouteDecision:
    """Q&A / VQA: retrieve every textual modality before answering.

    A question such as "ở đâu?" has no lexical ASR cue, but its answer is
    commonly spoken.  VQA therefore cannot rely on KIS's optional cue router:
    it searches visual, OCR, ASR, objects, and metadata for each query.
    """
    request = SearchRequest(
        query_id=query_id or _new_query_id("qa"),
        task="VQA",
        query_text=description,
        question=question,
        limit=limit,
    )
    return RouteDecision(request, ("visual", "ocr", "asr", "object", "metadata"))


_EVENT_SPLIT = re.compile(r"\(\d+\)\s*|;\s*|\n+")


def split_trake_events(query_text: str) -> list[str]:
    """Best-effort split of a TRAKE prompt into an ordered event list.

    Prefers an explicit "(1) ... (2) ... (3) ..." enumeration (as used in
    the contest's own example); falls back to splitting on ';' or newlines.
    Callers should still let a human/LLM re-split ambiguous cases — this is
    the rule-based fallback WP12 always has available.
    """
    parts = [p.strip(" .") for p in _EVENT_SPLIT.split(query_text) if p.strip(" .")]
    # Drop a leading clause like "Tìm N khoảnh khắc chính khi ..." if the
    # enumeration below it already carries the events.
    has_numbered_events = re.search(r"\(\d+\)", query_text) is not None
    if len(parts) > 1 and has_numbered_events and _looks_like_intro(parts[0]):
        parts = parts[1:]
    return parts


def _looks_like_intro(fragment: str) -> bool:
    return bool(re.match(r"^(tìm|find|xác định|liệt kê)\b", fragment, re.I)) and ":" in fragment or fragment.endswith(":")


def route_trake(query_text: str, events: list[str] | None = None, *, query_id: str | None = None, limit: int = 100) -> RouteDecision:
    ev = events or split_trake_events(query_text)
    request = SearchRequest(
        query_id=query_id or _new_query_id("trake"),
        task="TRAKE",
        query_text=query_text,
        events=tuple(ev),
        limit=limit,
    )
    return RouteDecision(request, _branches_for_text(query_text, include_metadata=False))
