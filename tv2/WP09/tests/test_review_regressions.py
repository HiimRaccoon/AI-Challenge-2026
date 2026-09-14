"""Regression coverage for the post-completion WP09 review."""

from __future__ import annotations

from pathlib import Path

import pytest

from wp09.config import RefinementConfig
from wp09.contracts import CoarseCandidate, ContractError, DecodeBudget, RefineRequest, RefinementContext, RefinementPolicy, Task
from wp09.decoder import DecodedFrame
from wp09.policies import ScoredFrame, select_hypotheses
from wp09.scoring import ScoringUnavailable, Siglip2Scorer
from wp09.service import ExactFrameRefiner
import wp09.cli as cli


CTX = RefinementContext("run-1", "media/1", "map-1", "decoder-1", "model-1", "config-1")
CONFIG = RefinementConfig(500, 2, 100, 24, 3, "fake", 1, "manual_only")


class CanonicalCountingDecoder:
    mapping_guaranteed = True

    def __init__(self) -> None:
        self.calls = 0
        self.requested: list[int | None] = []

    def duration_ms(self, video_path: Path) -> int:
        return 1_000

    def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None) -> tuple[DecodedFrame, ...]:
        self.calls += 1
        self.requested.append(max_frames)
        frames = tuple(DecodedFrame(n, n, "1/100", n * 10) for n in range(101) if start_ms <= n * 10 <= end_ms)
        return frames if max_frames is None else frames[:max_frames]


class Scores:
    model_name = "fake"
    model_version = "v1"
    def score(self, query_text: str, frames):
        return tuple(float(frame.frame_id) for frame in frames)


def request(budget: DecodeBudget = DecodeBudget(40, 1_000, None, 1)) -> RefineRequest:
    return RefineRequest(CoarseCandidate("L21_V001", 50, 500, confidence=0.8), Path("canonical.mp4"), Task.KIS, "door", RefinementPolicy.REPRESENTATIVE, CTX, budget)


def test_service_rejects_decoder_without_canonical_mapping_guarantee() -> None:
    """Catches an arbitrary decoder bypassing TV1's original-frame mapping."""
    class DuckDecoder:
        def duration_ms(self, video_path: Path) -> int: return 1_000
        def frames_between(self, *args): return ()
    with pytest.raises(ContractError, match="canonical mapping"):
        ExactFrameRefiner(DuckDecoder(), Scores(), CONFIG)


def test_sampler_passes_physical_frame_limits_to_decoder() -> None:
    """Catches collecting an unbounded decoder result then trimming it afterwards."""
    decoder = CanonicalCountingDecoder()
    result = ExactFrameRefiner(decoder, Scores(), CONFIG).refine(request(DecodeBudget(3, 1_000, None, 1)))
    assert decoder.requested and all(limit is not None for limit in decoder.requested)
    assert sum(limit for limit in decoder.requested if limit is not None) <= 3
    assert result.status == "partial"
    assert result.degraded_reason == "decode_budget_exhausted"


def test_cache_hit_avoids_a_second_decoder_call() -> None:
    """Catches a cache that is consulted only after original video has been decoded."""
    decoder = CanonicalCountingDecoder()
    refiner = ExactFrameRefiner(decoder, Scores(), CONFIG)
    refiner.refine(request())
    first = decoder.calls
    result = refiner.refine(request())
    assert result.audit.cache_hit is True
    assert decoder.calls == first


def test_siglip2_converts_runtime_oom_to_scorer_oom() -> None:
    """Catches GPU OOM falling through as a generic unavailable scorer."""
    class Processor:
        def __call__(self, **kwargs): return {}
    class Model:
        def __call__(self, **kwargs): raise RuntimeError("CUDA out of memory")
    class NoGrad:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Torch:
        def no_grad(self): return NoGrad()
    scorer = Siglip2Scorer()
    scorer._load = lambda: (Model(), Processor(), Torch())  # type: ignore[method-assign]
    with pytest.raises(ScoringUnavailable, match="scorer_oom") as error:
        scorer.score("door", (DecodedFrame(1, 1, "1/1000", 1, image_rgb=object()),))
    assert error.value.reason == "scorer_oom"


def _siglip2_fakes(*, oom_above: int | None = None):
    class Tensor:
        def __init__(self, values): self._values = values
        def reshape(self, *args): return self
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self._values
    class Processor:
        def __call__(self, *, images, **kwargs): return {"images": images}
    class Model:
        def __init__(self) -> None: self.batch_lengths: list[int] = []
        def __call__(self, *, images):
            self.batch_lengths.append(len(images))
            if oom_above is not None and len(images) > oom_above:
                raise RuntimeError("CUDA out of memory")
            return type("Output", (), {"logits_per_image": Tensor(images)})()
    class NoGrad:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Cuda:
        def __init__(self) -> None: self.empty_cache_calls = 0
        def empty_cache(self) -> None: self.empty_cache_calls += 1
    class Torch:
        def __init__(self) -> None: self.cuda = Cuda()
        def no_grad(self): return NoGrad()
    return Model(), Processor(), Torch()


def test_siglip2_retries_oom_batch_at_half_size_and_preserves_score_order() -> None:
    """Catches an OOM discarding a window instead of retrying the same frames smaller."""
    model, processor, torch = _siglip2_fakes(oom_above=2)
    scorer = Siglip2Scorer(batch_size=4)
    scorer._load = lambda: (model, processor, torch)  # type: ignore[method-assign]
    frames = tuple(DecodedFrame(index, index, "1/1000", index, image_rgb=index) for index in range(5))

    assert scorer.score("door", frames) == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert model.batch_lengths == [4, 2, 2, 1]
    assert torch.cuda.empty_cache_calls == 1


def test_siglip2_default_uses_eight_frame_batches() -> None:
    """Catches the configured throughput profile silently regressing to per-frame forwards."""
    model, processor, torch = _siglip2_fakes()
    scorer = Siglip2Scorer()
    scorer._load = lambda: (model, processor, torch)  # type: ignore[method-assign]
    frames = tuple(DecodedFrame(index, index, "1/1000", index, image_rgb=index) for index in range(9))

    assert scorer.score("door", frames) == tuple(float(index) for index in range(9))
    assert model.batch_lengths == [8, 1]


def test_vqa_evidence_is_counted_once_in_policy_score() -> None:
    """Catches evidence contribution being included in both composite and VQA boost."""
    from wp09.contracts import EvidenceContribution
    frame = ScoredFrame(DecodedFrame(1, 1, "1/1000", 1), visual_score=0.8, evidence=(EvidenceContribution("ocr", 0.2),))
    chosen = select_hypotheses(Task.VQA, RefinementPolicy.EVIDENCE_VISIBLE, (frame,), limit=1)
    assert chosen[0].policy_score == pytest.approx(1.0)


def test_cli_accepts_explicit_canonical_decoder_factory(monkeypatch, capsys) -> None:
    """Catches CLI rejecting a mapped decoder supplied by the integration adapter."""
    decoder = CanonicalCountingDecoder()
    monkeypatch.setattr(cli, "request_from_dict", lambda payload: request())
    monkeypatch.setattr(cli.json, "loads", lambda text: {})
    monkeypatch.setattr(cli, "_load_factory", lambda spec: (lambda *args: decoder))
    assert cli.main(["refine", "--request", "configs/default.yaml", "--config", "configs/default.yaml", "--decoder-factory", "team:canonical"]) == 0
    assert '"status": "manual_only"' in capsys.readouterr().out


def test_cli_rejects_noncanonical_decoder_factory(monkeypatch, capsys) -> None:
    """Catches CLI accepting a factory that returns undecidable proxy-frame IDs."""
    class DuckDecoder:
        def duration_ms(self, video_path: Path) -> int: return 1_000
        def frames_between(self, *args, **kwargs): return ()
    monkeypatch.setattr(cli, "request_from_dict", lambda payload: request())
    monkeypatch.setattr(cli.json, "loads", lambda text: {})
    monkeypatch.setattr(cli, "_load_factory", lambda spec: (lambda *args: DuckDecoder()))
    assert cli.main(["refine", "--request", "configs/default.yaml", "--config", "configs/default.yaml", "--decoder-factory", "team:duck"]) == 2
    assert "canonical mapping" in capsys.readouterr().err


def test_pyav_seeks_and_counts_warmup_against_physical_frame_budget(monkeypatch) -> None:
    """Catches PyAV decoding an unbounded stream prefix before a local window."""
    from fractions import Fraction
    import wp09.pyav_decoder as pyav_decoder

    class Frame:
        def __init__(self, pts: int) -> None: self.pts = pts
        def to_ndarray(self, format: str): return object()
    class Stream:
        duration = 1_000
        time_base = Fraction(1, 1_000)
    class Container:
        def __init__(self) -> None:
            self.streams = type("Streams", (), {"video": [Stream()]})()
            self.seek_offset = None
            self.decoded = 0
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def seek(self, offset, **kwargs): self.seek_offset = offset
        def decode(self, stream):
            for pts in (450, 500, 550, 600):
                self.decoded += 1
                yield Frame(pts)
    container = Container()
    monkeypatch.setattr(pyav_decoder, "_load_av", lambda: __import__("types").SimpleNamespace(open=lambda path: container))
    frames = pyav_decoder.PyAVVideoDecoder().raw_frames_between(Path("original.mp4"), 500, 700, 24, max_frames=2)
    assert container.seek_offset == 500
    assert container.decoded == 2
    assert [frame.pts for frame in frames] == [500]


def test_siglip2_load_converts_oom_to_scorer_oom(monkeypatch) -> None:
    """Catches model allocation OOM being misreported as a generic unavailable scorer."""
    import sys
    import types
    torch = types.ModuleType("torch")
    transformers = types.ModuleType("transformers")
    class Processor:
        @staticmethod
        def from_pretrained(name): return object()
    class Model:
        @staticmethod
        def from_pretrained(name): raise RuntimeError("CUDA out of memory during load")
    transformers.AutoModel = Model
    transformers.AutoProcessor = Processor
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(ScoringUnavailable, match="scorer_oom") as error:
        Siglip2Scorer()._load()
    assert error.value.reason == "scorer_oom"


def test_warmup_budget_exhaustion_raises_unavailable_instead_of_manual_only() -> None:
    """Catches a non-decoded coarse candidate being presented as a manual original frame."""
    from wp09.decoder import DecodeBudgetExhausted
    from wp09.contracts import RefinementUnavailable
    class WarmupExhaustedDecoder:
        mapping_guaranteed = True
        def duration_ms(self, video_path: Path) -> int: return 1_000
        def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None):
            raise DecodeBudgetExhausted()
    with pytest.raises(RefinementUnavailable, match="decode_budget_exhausted"):
        ExactFrameRefiner(WarmupExhaustedDecoder(), Scores(), CONFIG).refine(request(DecodeBudget(2, 1_000, None, 1)))


def test_pyav_reports_warmup_exhaustion_before_any_local_frame(monkeypatch) -> None:
    """Catches PyAV returning an empty window that service mistakes for decode failure."""
    from fractions import Fraction
    from wp09.decoder import DecodeBudgetExhausted
    import wp09.pyav_decoder as pyav_decoder
    class Frame:
        def __init__(self, pts): self.pts = pts
        def to_ndarray(self, format): return object()
    class Stream:
        duration = 1_000
        time_base = Fraction(1, 1_000)
    class Container:
        streams = type("Streams", (), {"video": [Stream()]})()
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def seek(self, *args, **kwargs): pass
        def decode(self, stream):
            yield Frame(100)
            yield Frame(200)
    monkeypatch.setattr(pyav_decoder, "_load_av", lambda: __import__("types").SimpleNamespace(open=lambda path: Container()))
    with pytest.raises(DecodeBudgetExhausted):
        pyav_decoder.PyAVVideoDecoder().raw_frames_between(Path("original.mp4"), 500, 700, 24, max_frames=2)


def test_pyav_fps_gap_counts_against_physical_frame_budget(monkeypatch) -> None:
    """Catches skipped near-duplicate frames bypassing max_frames until stream end."""
    from fractions import Fraction
    import wp09.pyav_decoder as pyav_decoder
    class Frame:
        def __init__(self, pts): self.pts = pts
        def to_ndarray(self, format): return object()
    class Stream:
        duration = 1_000
        time_base = Fraction(1, 1_000)
    class Container:
        streams = type("Streams", (), {"video": [Stream()]})()
        def __init__(self): self.decoded = 0
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def seek(self, *args, **kwargs): pass
        def decode(self, stream):
            for pts in (500, 510, 520):
                self.decoded += 1
                yield Frame(pts)
    container = Container()
    monkeypatch.setattr(pyav_decoder, "_load_av", lambda: __import__("types").SimpleNamespace(open=lambda path: container))
    frames = pyav_decoder.PyAVVideoDecoder().raw_frames_between(Path("original.mp4"), 500, 700, 1, max_frames=2)
    assert [frame.pts for frame in frames] == [500]
    assert container.decoded == 2
