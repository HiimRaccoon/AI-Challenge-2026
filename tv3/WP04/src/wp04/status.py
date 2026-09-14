from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal


State = Literal["ready", "no_audio", "failed", "disabled"]


@dataclass(frozen=True, slots=True)
class ModalityStatus:
    video_id: str
    modality: Literal["ocr", "asr", "object", "metadata"]
    fingerprint: str
    state: State
    error_message: str | None = None
    preprocess_run_id: str | None = None
    wp04_artifact_set_id: str | None = None

    @classmethod
    def ready(cls, video_id: str, modality: str, fingerprint: str, **provenance: str) -> "ModalityStatus":
        return cls(video_id, modality, fingerprint, "ready", **provenance)  # type: ignore[arg-type]

    @classmethod
    def no_audio(cls, video_id: str, modality: str, fingerprint: str, **provenance: str) -> "ModalityStatus":
        return cls(video_id, modality, fingerprint, "no_audio", **provenance)  # type: ignore[arg-type]

    @classmethod
    def failed(cls, video_id: str, modality: str, fingerprint: str, error: str, **provenance: str) -> "ModalityStatus":
        return cls(video_id, modality, fingerprint, "failed", error, **provenance)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id, "modality": self.modality, "fingerprint": self.fingerprint,
            "state": self.state, "error_message": self.error_message,
            "preprocess_run_id": self.preprocess_run_id,
            "wp04_artifact_set_id": self.wp04_artifact_set_id,
        }


def should_skip(status: ModalityStatus | None, fingerprint: str) -> bool:
    return status is not None and status.fingerprint == fingerprint and status.state in {"ready", "no_audio"}
