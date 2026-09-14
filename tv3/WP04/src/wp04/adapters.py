"""Model seams. Runtime packages are imported only by explicit configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from .contracts import AudioRecord, FrameRecord


class AdapterUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OCRRawDetection:
    text: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class ASRRawSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ObjectRawDetection:
    label: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    confidence: float


class OCRAdapter(Protocol):
    model_name: str
    model_version: str

    def detect(self, frame: FrameRecord) -> Sequence[Any]: ...


class ASRAdapter(Protocol):
    model_name: str
    model_version: str
    vad_version: str

    def transcribe(self, audio: AudioRecord) -> Sequence[Any]: ...


class ObjectAdapter(Protocol):
    model_name: str
    model_version: str

    def detect(self, frame: FrameRecord) -> Sequence[Any]: ...


def load_adapter(factory_path: str, **settings: Any) -> Any:
    """Load a configured adapter without importing optional models at package import time."""
    try:
        module_name, attribute = factory_path.rsplit(":", 1)
        factory = getattr(import_module(module_name), attribute)
        return factory(**settings)
    except (ImportError, AttributeError, ValueError, TypeError) as error:
        raise AdapterUnavailableError(f"adapter unavailable: {factory_path}: {error}") from error
