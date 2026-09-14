"""Deterministic TRAKE event-state parser used by sequence verification."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Sequence


def _fold(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


_PRONOUNS = {"he", "she", "they", "him", "her", "anh", "chi", "ong", "ba", "ho"}
_ENTITY_TERMS = {
    "man", "woman", "person", "boy", "girl", "employee", "customer", "nguoi",
    "nam", "nu", "nhan vien", "khach", "tre em",
}
_OBJECT_TERMS = {
    "product", "item", "bag", "box", "bottle", "phone", "book", "car", "bike",
    "san pham", "tui", "hop", "chai", "dien thoai", "sach", "xe",
}
_LOCATION_TERMS = {
    "shop", "store", "room", "street", "house", "kitchen", "stage", "door",
    "cua hang", "phong", "duong", "nha", "bep", "san khau", "cua",
}
_HOLD_ACTIONS = {
    "pick", "picks", "picked", "take", "takes", "took", "hold", "holds",
    "held", "carry", "carries", "carried", "cầm", "nhặt", "lấy", "nắm",
}
_RELEASE_ACTIONS = {
    "put", "puts", "drop", "drops", "dropped", "give", "gives", "gave",
    "đặt", "thả", "đưa",
}
_MOVE_ACTIONS = {
    "enter", "enters", "entered", "leave", "leaves", "left", "walk", "walks",
    "walked", "run", "runs", "ran", "đi", "chạy", "bước",
}
_OTHER_ACTIONS = {
    "trộn", "cắt", "xếp", "ngậm", "treo", "ngoảnh", "đứng", "co",
    "rót", "đổ", "khuấy", "cuộn", "nắn", "bơi", "mở", "đóng",
    "mix", "mixes", "cut", "cuts", "pour", "pours", "turn", "turns",
    "hang", "hangs", "stand", "stands", "bite", "bites",
}


def _terms(text: str, vocabulary: set[str]) -> tuple[str, ...]:
    folded = _fold(text)
    return tuple(sorted(
        term for term in vocabulary
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded)
    ))


@dataclass(frozen=True)
class EventState:
    event_index: int
    text: str
    entities: tuple[str, ...]
    objects: tuple[str, ...]
    locations: tuple[str, ...]
    actions: tuple[str, ...]
    refers_to_previous_entity: bool
    state_transition: str | None
    temporal_cues: tuple[str, ...] = ()
    ordinal: int | None = None


def parse_event(event_index: int, text: str) -> EventState:
    folded = _fold(text)
    words = set(re.findall(r"[a-z0-9]+", folded))
    # Accent folding merges fruit "cam" with "cầm" and superlative "nhất"
    # with "nhặt". Preserve the original spelling for action recognition.
    action_words = set(re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text.casefold())))
    actions = tuple(sorted(action_words & (_HOLD_ACTIONS | _RELEASE_ACTIONS | _MOVE_ACTIONS | _OTHER_ACTIONS)))
    transition = None
    if action_words & _HOLD_ACTIONS:
        transition = "holding"
    elif action_words & _RELEASE_ACTIONS:
        transition = "released"
    elif action_words & _MOVE_ACTIONS:
        transition = "moved"
    cues = tuple(name for name, pattern in (
        ("start", r"\b(bat dau|begins?|starts?)\b"),
        ("completion", r"\b(hoan tat|hoan thanh|completes?|finishes?)\b"),
        ("first", r"\b(dau tien|first)\b"),
        ("last", r"\b(cuoi cung|last|final)\b"),
    ) if re.search(pattern, folded))
    ordinal_match = re.search(r"\bthu\s+(\d+|hai|ba|tu|nam|sau|bay|tam|chin|muoi)\b|\b(\d+)(?:st|nd|rd|th)\b", folded)
    ordinal = 1 if "first" in cues else None
    if ordinal_match:
        value = ordinal_match.group(1) or ordinal_match.group(2)
        ordinal = int(value) if value.isdigit() else {
            "hai": 2, "ba": 3, "tu": 4, "nam": 5, "sau": 6,
            "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
        }[value]
    return EventState(
        event_index=event_index,
        text=text,
        entities=_terms(text, _ENTITY_TERMS),
        objects=_terms(text, _OBJECT_TERMS),
        locations=_terms(text, _LOCATION_TERMS),
        actions=actions,
        refers_to_previous_entity=bool(words & _PRONOUNS),
        state_transition=transition,
        temporal_cues=cues,
        ordinal=ordinal,
    )


def parse_event_graph(events: Sequence[str]) -> tuple[EventState, ...]:
    return tuple(parse_event(index, text) for index, text in enumerate(events))
