"""Dependency-free exact-frame request state helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CertifiedStepRequest:
    """One UI step expressed relative to a persistent certified anchor."""

    anchor_frame_id: int
    anchor_timestamp_ms: int
    effective_offsets: tuple[int, ...]


def certified_step_request(
    frame_id: int, timestamp_ms: int, offsets: Sequence[int], *,
    certified_anchor_frame_id: int | None = None, certified_anchor_timestamp_ms: int | None = None,
    cumulative_offset: int = 0,
) -> CertifiedStepRequest:
    if (certified_anchor_frame_id is None) != (certified_anchor_timestamp_ms is None):
        raise ValueError("certified anchor frame_id and timestamp_ms must be supplied together")
    anchor_frame_id = certified_anchor_frame_id if certified_anchor_frame_id is not None else frame_id
    anchor_timestamp_ms = certified_anchor_timestamp_ms if certified_anchor_timestamp_ms is not None else timestamp_ms
    return CertifiedStepRequest(anchor_frame_id, anchor_timestamp_ms, tuple(cumulative_offset + offset for offset in offsets))
