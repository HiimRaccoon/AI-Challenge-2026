"""Contract tests for the WP09 completion requirements."""

from __future__ import annotations

from pathlib import Path

import pytest

from wp09.contracts import (
    CoarseCandidate,
    DecodeBudget,
    EvidenceContribution,
    RefineRequest,
    RefinementContext,
    RefinementUnavailable,
    RefinementPolicy,
    Task,
)
from wp09.decoder import DecodedFrame
from wp09.service import ExactFrameRefiner


CONTEXT = RefinementContext("run-7", "media/L21_V001", "map-v3", "decoder-1", "model-1", "config-1")
BUDGET = DecodeBudget(8, 1_000, None, 1)


class BrokenDecoder:
    mapping_guaranteed = True
    def duration_ms(self, video_path: Path) -> int:
        raise RuntimeError("source missing")

    def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None) -> tuple[DecodedFrame, ...]:
        raise RuntimeError("source missing")


def test_request_carries_immutable_context_budget_and_optional_evidence() -> None:
    """Catches losing the upstream mapping run needed to reproduce a true frame."""

    request = RefineRequest(
        candidate=CoarseCandidate("L21_V001", 50, 500, upstream_score=0.8, confidence=0.2),
        video_path=Path("canonical-media.mp4"), task=Task.VQA, refinement_text="door colour",
        policy=RefinementPolicy.EVIDENCE_VISIBLE, context=CONTEXT, decode_budget=BUDGET,
        evidence=(EvidenceContribution("ocr", 0.2, "red"),),
    )

    assert request.context.mapping_ref == "map-v3"
    assert request.decode_budget.max_decoded_frames == 8
    assert request.evidence[0].source == "ocr"


def test_fatal_original_video_decode_is_not_fabricated_as_manual_only() -> None:
    """Catches masking inaccessible original media as an operator-selectable result."""

    config = _config()
    request = _request()
    with pytest.raises(RefinementUnavailable, match="decode_failure"):
        ExactFrameRefiner(BrokenDecoder(), None, config).refine(request)


def _request() -> RefineRequest:
    return RefineRequest(
        CoarseCandidate("L21_V001", 50, 500, confidence=0.8), Path("canonical-media.mp4"), Task.KIS,
        "person opens door", RefinementPolicy.REPRESENTATIVE, context=CONTEXT, decode_budget=BUDGET,
    )


def _config():
    from wp09.config import RefinementConfig

    return RefinementConfig(500, 2, 100, 24, 3, "fake", 1, "manual_only")
