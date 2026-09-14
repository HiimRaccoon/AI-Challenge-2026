from __future__ import annotations

from pathlib import Path

from wp09.config import RefinementConfig
from wp09.contracts import CoarseCandidate, DecodeBudget, RefineRequest, RefinementContext, RefinementPolicy, Task
from wp09.decoder import DecodedFrame
from wp09.service import ExactFrameRefiner


class FakeDecoder:
    mapping_guaranteed = True
    def duration_ms(self, video_path: Path) -> int:
        return 1_000

    def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None) -> tuple[DecodedFrame, ...]:
        frames = tuple(DecodedFrame(frame, frame, "1/100", frame * 10) for frame in range(101) if start_ms <= frame * 10 <= end_ms)
        return frames if max_frames is None else frames[:max_frames]


class FailingScorer:
    model_name = "missing"
    model_version = "none"

    def score(self, query_text: str, frames: tuple[DecodedFrame, ...]) -> tuple[float, ...]:
        raise RuntimeError("model unavailable")


class ConstantScorer:
    model_name = "fake"
    model_version = "v1"

    def score(self, query_text: str, frames: tuple[DecodedFrame, ...]) -> tuple[float, ...]:
        return tuple(float(frame.frame_id) for frame in frames)


CONTEXT = RefinementContext("run-1", "media/1", "map-1", "decoder-1", "model-1", "config-1")
BUDGET = DecodeBudget(200, 1_000, None, 1)


def test_refiner_preserves_coarse_candidate_when_scoring_is_unavailable() -> None:
    """Catches a model outage removing the operator's original candidate/window."""

    config = RefinementConfig(500, 2, 100, 24, 3, "fake", 1, "manual_only")
    request = RefineRequest(
        candidate=CoarseCandidate("L21_V001", 50, 500), video_path=Path("L21_V001.mp4"),
        task=Task.KIS, refinement_text="person opens a door", policy=RefinementPolicy.REPRESENTATIVE, context=CONTEXT, decode_budget=BUDGET,
    )

    result = ExactFrameRefiner(FakeDecoder(), FailingScorer(), config).refine(request)

    assert result.coarse_candidate == request.candidate
    assert result.degraded_reason == "scorer_unavailable"
    assert result.hypotheses[0].frame_id == 50
    assert result.status == "manual_only"


def test_visual_only_scoring_is_refined_when_optional_evidence_is_absent() -> None:
    """Catches treating missing OCR/ASR/object evidence as a scorer failure."""

    config = RefinementConfig(500, 2, 100, 24, 3, "fake", 1, "manual_only")
    request = RefineRequest(CoarseCandidate("L21_V001", 50, 500), Path("L21_V001.mp4"), Task.KIS, "door", RefinementPolicy.REPRESENTATIVE, CONTEXT, BUDGET)

    result = ExactFrameRefiner(FakeDecoder(), ConstantScorer(), config).refine(request)

    assert result.status == "refined"
    assert result.degraded_reason is None
