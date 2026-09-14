"""Optional visual scoring; imports CUDA/model packages only when scoring runs."""

from __future__ import annotations

from typing import Protocol, Sequence

from .decoder import DecodedFrame


class ScoringUnavailable(RuntimeError):
    """The original frames are available but automated visual scoring is not."""

    def __init__(self, reason: str = "scorer_unavailable") -> None:
        self.reason = reason
        super().__init__(reason)


class FrameScorer(Protocol):
    model_name: str
    model_version: str
    def score(self, query_text: str, frames: Sequence[DecodedFrame]) -> tuple[float, ...]: ...


class Siglip2Scorer:
    """Lazy one-model SIGLIP2 adapter; no import/download occurs at construction."""

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", *, batch_size: int = 8) -> None:
        self.model_name = model_name
        self.model_version = model_name
        self._batch_size = batch_size
        self._model: object | None = None
        self._processor: object | None = None

    def _load(self) -> tuple[object, object, object]:
        if self._model is None or self._processor is None:
            try:
                import torch
                from transformers import AutoModel, AutoProcessor
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
            except Exception as exc:
                raise ScoringUnavailable(_failure_reason(exc)) from exc
        try:
            import torch
        except Exception as exc:
            raise ScoringUnavailable("scorer_unavailable") from exc
        return self._model, self._processor, torch

    def score(self, query_text: str, frames: Sequence[DecodedFrame]) -> tuple[float, ...]:
        if not frames or any(frame.image_rgb is None for frame in frames):
            raise ScoringUnavailable("scorer_unavailable")
        model, processor, torch = self._load()
        scores: list[float] = []
        offset = 0
        active_batch_size = self._batch_size
        while offset < len(frames):
            images = [frame.image_rgb for frame in frames[offset: offset + active_batch_size]]
            try:
                inputs = processor(text=[query_text] * len(images), images=images, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits_per_image.reshape(-1).detach().cpu().tolist()
                scores.extend(float(value) for value in logits)
                offset += len(images)
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise ScoringUnavailable("scorer_unavailable") from exc
                if active_batch_size == 1:
                    raise ScoringUnavailable("scorer_oom") from exc
                _empty_cuda_cache(torch)
                active_batch_size = max(1, active_batch_size // 2)
            except Exception as exc:
                raise ScoringUnavailable("scorer_unavailable") from exc
        return tuple(scores)


def _failure_reason(error: BaseException) -> str:
    return "scorer_oom" if _is_oom(error) else "scorer_unavailable"


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def _empty_cuda_cache(torch: object) -> None:
    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:
            pass
