"""Object-detection enrichment: canonical label + spatial region.

Ported from TV1_TV3_WP04/src/aic2026/objects.py (_canonical_label,
_spatial_region, _alias_map) with the Settings/pydantic dependency removed —
alias overrides are now a plain dict so this has zero coupling to that repo.
"""
from __future__ import annotations

import unicodedata

_DEFAULT_ALIASES: dict[str, list[str]] = {
    "person": ["person", "người", "nguoi"],
    "car": ["car", "xe hơi", "xe hoi", "ô tô", "o to"],
    "motorcycle": ["motorcycle", "xe máy", "xe may"],
    "laptop": ["laptop", "máy tính xách tay", "may tinh xach tay"],
    "book": ["book", "sách", "sach"],
    "chair": ["chair", "ghế", "ghe"],
    "table": ["table", "dining table", "bàn", "ban"],
    "scale": [
        "scale", "weighing scale", "digital scale", "electronic scale",
        "cân điện tử", "can dien tu", "cái cân", "cai can", "chiếc cân", "chiec can",
    ],
    "dog": ["dog", "chó", "cho"],
    "cat": ["cat", "mèo", "meo"],
}


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def build_alias_map(extra_aliases: dict[str, list[str]] | None = None) -> dict[str, str]:
    """label variant (normalized, diacritics-insensitive) -> canonical label."""
    merged = dict(_DEFAULT_ALIASES)
    if extra_aliases:
        merged.update({str(k): list(v) for k, v in extra_aliases.items()})
    aliases: dict[str, str] = {}
    for canonical, variants in merged.items():
        canonical_norm = _normalize_label(canonical)
        aliases[canonical_norm] = canonical_norm
        aliases[_normalize_label(_strip_diacritics(canonical_norm))] = canonical_norm
        for variant in variants:
            variant_norm = _normalize_label(variant)
            aliases[variant_norm] = canonical_norm
            aliases[_normalize_label(_strip_diacritics(variant_norm))] = canonical_norm
    return aliases


def canonical_label(label: str, alias_map: dict[str, str]) -> tuple[str, list[str]]:
    """Returns (canonical_label, sorted list of known alias strings for it)."""
    normalized = _normalize_label(label)
    canonical = (
        alias_map.get(normalized)
        or alias_map.get(_normalize_label(_strip_diacritics(normalized)))
        or normalized
    )
    known_aliases = sorted({key for key, value in alias_map.items() if value == canonical})
    return canonical, known_aliases


def spatial_region(bbox_xyxy_norm: tuple[float, float, float, float]) -> str:
    """3x3 grid region name from a normalized [x1,y1,x2,y2] box, e.g. 'top_left', 'center'."""
    x1, y1, x2, y2 = bbox_xyxy_norm
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
    vertical = "top" if center_y < 1 / 3 else "bottom" if center_y > 2 / 3 else "center"
    if vertical == "center" and horizontal == "center":
        return "center"
    if vertical == "center":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}_{horizontal}"
