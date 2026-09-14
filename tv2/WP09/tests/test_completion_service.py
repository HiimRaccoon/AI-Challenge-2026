from __future__ import annotations

from pathlib import Path

from wp09.config import RefinementConfig
from wp09.contracts import CoarseCandidate, DecodeBudget, RefineRequest, RefinementContext, RefinementPolicy, Task
from wp09.decoder import DecodedFrame
from wp09.service import ExactFrameRefiner


CONTEXT = RefinementContext("run-7", "media/L21_V001", "map-v3", "decoder-1", "model-1", "config-1")


class FramesDecoder:
    mapping_guaranteed = True
    def duration_ms(self, video_path: Path) -> int:
        return 1_000

    def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None) -> tuple[DecodedFrame, ...]:
        frames = tuple(DecodedFrame(n, n, "1/100", n * 10) for n in range(101) if start_ms <= n * 10 <= end_ms)
        return frames if max_frames is None else frames[:max_frames]


class Scores:
    model_name = "fake"
    model_version = "v1"
    def score(self, query_text: str, frames: tuple[DecodedFrame, ...]) -> tuple[float, ...]:
        return tuple(float(frame.frame_id) for frame in frames)


def _config() -> RefinementConfig:
    return RefinementConfig(500, 2, 100, 24, 3, "fake", 1, "manual_only")


def _request(*, confidence: float = 0.8, budget: DecodeBudget | None = None) -> RefineRequest:
    return RefineRequest(CoarseCandidate("L21_V001", 50, 500, upstream_score=0.6, confidence=confidence), Path("canonical.mp4"), Task.KIS, "door", RefinementPolicy.REPRESENTATIVE, context=CONTEXT, decode_budget=budget or DecodeBudget(200, 1_000, None, 1))


def test_result_preserves_context_in_result_and_audit() -> None:
    """Catches a result that cannot identify the mapping which resolved its frames."""
    result = ExactFrameRefiner(FramesDecoder(), Scores(), _config()).refine(_request())
    assert result.context.preprocess_run_id == "run-7"
    assert result.audit.context.mapping_ref == "map-v3"
    assert result.audit.before_frame_id == 50
    assert result.audit.after_frame_id == result.hypotheses[0].frame_id


def test_manual_only_keeps_coarse_frame_as_selectable_hypothesis() -> None:
    """Catches model fallback erasing the original coarse candidate from manual review."""
    result = ExactFrameRefiner(FramesDecoder(), None, _config()).refine(_request())
    assert result.status == "manual_only"
    assert 50 in {item.frame_id for item in result.hypotheses}


def test_budget_exhaustion_yields_partial_when_trustworthy_scores_exist() -> None:
    """Catches budget clipping being presented as a fully executed refinement."""
    result = ExactFrameRefiner(FramesDecoder(), Scores(), _config()).refine(_request(budget=DecodeBudget(2, 1_000, None, 1)))
    assert result.status == "partial"
    assert result.degraded_reason == "decode_budget_exhausted"


def test_low_confidence_receives_a_wider_bounded_window() -> None:
    """Catches confidence being recorded but ignored by local-search radius selection."""
    refiner = ExactFrameRefiner(FramesDecoder(), Scores(), _config())
    assert refiner.radius_for(_request(confidence=0.1)) > refiner.radius_for(_request(confidence=0.9))


def test_second_identical_resolved_window_records_cache_hit() -> None:
    """Catches an internal cache that stores frames but is never consumed."""
    refiner = ExactFrameRefiner(FramesDecoder(), Scores(), _config())
    refiner.refine(_request())
    assert refiner.refine(_request()).audit.cache_hit is True
