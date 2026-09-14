from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedText:
    canonical: str
    without_diacritics: str


def normalize_text(text: str) -> NormalizedText:
    canonical = unicodedata.normalize("NFC", text).strip().lower()
    decomposed = unicodedata.normalize("NFD", canonical)
    folded = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    folded = folded.replace("đ", "d")
    return NormalizedText(canonical, unicodedata.normalize("NFC", folded))
