"""Independent modality orchestration for WP04."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import sys
import time
from typing import Any, Protocol, Sequence

from .adapters import ASRAdapter, OCRAdapter, ObjectAdapter
from .adapters import ASRRawSegment, OCRRawDetection, ObjectRawDetection
from .contracts import ASRSegment, AudioRecord, FrameRecord, OCRDetection, ObjectDetection
from .evidence import assign_ocr_evidence_ids
from .metadata import build_metadata
from .normalization import normalize_text
from .status import ModalityStatus
from .storage import ArtifactStore
from .temporal import LocalTemporalResolver, TemporalResolver


class ModalityAdapter(Protocol):
    """Small seam used by the pipeline before production adapters are wired."""

    def run(self, video_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class FakeAdapter:
    """Deterministic adapter for pipeline tests and CPU-only development."""

    error: str | None = None

    def run(self, video_id: str) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)


class Pipeline:
    """Run every requested modality independently for every video."""

    def __init__(self, adapters: Mapping[str, ModalityAdapter]) -> None:
        self._adapters = dict(adapters)

    def run(self, video_ids: Iterable[str], fingerprint: str) -> list[ModalityStatus]:
        statuses: list[ModalityStatus] = []
        for video_id in video_ids:
            for modality, adapter in self._adapters.items():
                try:
                    adapter.run(video_id)
                except Exception as error:
                    statuses.append(
                        ModalityStatus.failed(video_id, modality, fingerprint, str(error))
                    )
                else:
                    statuses.append(ModalityStatus.ready(video_id, modality, fingerprint))
        return statuses


@dataclass(frozen=True, slots=True)
class PipelineResult:
    statuses: tuple[ModalityStatus, ...]

    def status_for(self, video_id: str, modality: str) -> ModalityStatus:
        for status in self.statuses:
            if (status.video_id, status.modality) == (video_id, modality):
                return status
        raise KeyError(f"no status for {video_id}/{modality}")


class WP04Pipeline:
    """Runs OCR, ASR, objects and metadata independently for each TV1 video."""

    def __init__(
        self, ocr: OCRAdapter, asr: ASRAdapter, objects: ObjectAdapter,
        metadata: Callable[[str], None] | None = None, *, enabled: Mapping[str, bool] | None = None,
        ocr_batch_size: int = 1, object_batch_size: int = 1,
        ocr_sample_interval_ms: int = 0, object_sample_interval_ms: int = 0,
    ) -> None:
        if ocr_batch_size < 1 or object_batch_size < 1:
            raise ValueError("OCR and object batch sizes must be positive")
        if ocr_sample_interval_ms < 0 or object_sample_interval_ms < 0:
            raise ValueError("OCR and object sample intervals must be non-negative")
        self._ocr, self._asr, self._objects, self._metadata = ocr, asr, objects, metadata
        self._ocr_batch_size, self._object_batch_size = ocr_batch_size, object_batch_size
        self._ocr_sample_interval_ms = ocr_sample_interval_ms
        self._object_sample_interval_ms = object_sample_interval_ms
        configured = dict(enabled or {})
        self._enabled = {
            modality: bool(configured.get(modality, True))
            for modality in ("ocr", "asr", "object", "metadata")
        }

    @staticmethod
    def _batches(items: Sequence[FrameRecord], batch_size: int) -> Iterable[Sequence[FrameRecord]]:
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    @staticmethod
    def _sample_frames(
        items: Sequence[FrameRecord], sample_interval_ms: int,
    ) -> tuple[FrameRecord, ...]:
        """Select temporal samples while retaining video and shot boundaries."""
        if not items or sample_interval_ms == 0:
            return tuple(items)
        selected_indices = {0, len(items) - 1}
        last_timestamp_ms = items[0].timestamp_ms
        for index, frame in enumerate(items[1:], start=1):
            if frame.timestamp_ms - last_timestamp_ms >= sample_interval_ms:
                selected_indices.add(index)
                last_timestamp_ms = frame.timestamp_ms
        shots: dict[str, list[int]] = defaultdict(list)
        for index, frame in enumerate(items):
            if frame.shot_id:
                shots[frame.shot_id].append(index)
        for indices in shots.values():
            selected_indices.add(indices[0])
            selected_indices.add(indices[-1])
        return tuple(items[index] for index in sorted(selected_indices))

    @staticmethod
    def _detect_batch(adapter: object, frames: Sequence[FrameRecord], expected: type[object], modality: str) -> list[list[object]]:
        detect_batch = getattr(adapter, "detect_batch", None)
        raw_by_frame = detect_batch(frames) if callable(detect_batch) else [adapter.detect(frame) for frame in frames]
        if not isinstance(raw_by_frame, Sequence) or len(raw_by_frame) != len(frames):
            raise TypeError(f"{modality} batch adapter returned an unexpected number of results")
        return [WP04Pipeline._require_raw(raw, expected, modality) for raw in raw_by_frame]

    @staticmethod
    def _run(operation: Callable[[], object], video_id: str, modality: str, fingerprint: str) -> ModalityStatus:
        try:
            operation()
        except Exception as error:
            return ModalityStatus.failed(video_id, modality, fingerprint, str(error))
        return ModalityStatus.ready(video_id, modality, fingerprint)

    def run(
        self, frames: Iterable[FrameRecord], audio_by_video: Mapping[str, AudioRecord], fingerprint: str,
    ) -> PipelineResult:
        grouped: dict[str, list[FrameRecord]] = defaultdict(list)
        for frame in frames:
            grouped[frame.video_id].append(frame)
        statuses: list[ModalityStatus] = []
        for video_id in sorted(grouped):
            video_frames = tuple(sorted(grouped[video_id], key=lambda frame: (frame.timestamp_ms, frame.frame_id)))
            ocr_frames = self._sample_frames(video_frames, self._ocr_sample_interval_ms)
            object_frames = self._sample_frames(video_frames, self._object_sample_interval_ms)
            if self._enabled["ocr"]:
                statuses.append(self._run(
                    lambda: [self._detect_batch(self._ocr, batch, OCRRawDetection, "ocr")
                             for batch in self._batches(ocr_frames, self._ocr_batch_size)],
                    video_id, "ocr", fingerprint,
                ))
            else:
                statuses.append(ModalityStatus(video_id, "ocr", fingerprint, "disabled"))
            audio = audio_by_video.get(video_id)
            if not self._enabled["asr"]:
                statuses.append(ModalityStatus(video_id, "asr", fingerprint, "disabled"))
            elif audio is None:
                statuses.append(ModalityStatus.failed(video_id, "asr", fingerprint, "expected audio record missing"))
            elif not audio.declared_present:
                statuses.append(ModalityStatus.no_audio(video_id, "asr", fingerprint))
            else:
                statuses.append(self._run(lambda: self._asr.transcribe(audio), video_id, "asr", fingerprint))
            if self._enabled["object"]:
                statuses.append(self._run(
                    lambda: [self._detect_batch(self._objects, batch, ObjectRawDetection, "object")
                             for batch in self._batches(object_frames, self._object_batch_size)],
                    video_id, "object", fingerprint,
                ))
            else:
                statuses.append(ModalityStatus(video_id, "object", fingerprint, "disabled"))
            if not self._enabled["metadata"] or self._metadata is None:
                statuses.append(ModalityStatus(video_id, "metadata", fingerprint, "disabled"))
            else:
                statuses.append(self._run(lambda: self._metadata(video_id), video_id, "metadata", fingerprint))
        return PipelineResult(tuple(statuses))

    @staticmethod
    def _model(adapter: object, fallback_name: str) -> tuple[str, str]:
        return (str(getattr(adapter, "model_name", fallback_name)), str(getattr(adapter, "model_version", "unknown")))

    @staticmethod
    def _require_raw(values: Iterable[object], expected: type[object], modality: str) -> list[object]:
        materialized = list(values)
        if any(not isinstance(value, expected) for value in materialized):
            raise TypeError(f"{modality} adapter returned an unsupported raw record")
        return materialized

    @staticmethod
    def _stored_status(store: ArtifactStore, video_id: str, modality: str) -> ModalityStatus:
        status = store.status_for(video_id, modality)
        if status is None:
            raise RuntimeError(f"missing persisted status for {video_id}/{modality}")
        return status

    def _completed_video_statuses(
        self, store: ArtifactStore, video_id: str, fingerprint: str,
    ) -> tuple[ModalityStatus, ...] | None:
        statuses: list[ModalityStatus] = []
        for modality in ("ocr", "asr", "object", "metadata"):
            status = store.status_for(video_id, modality)
            if self._enabled[modality]:
                if not store.completed(video_id, modality, fingerprint):
                    return None
            elif status is None or status.fingerprint != fingerprint or status.state != "disabled":
                return None
            statuses.append(status)
        return tuple(statuses)

    def run_and_store(
        self, frames: Iterable[FrameRecord], audio_by_video: Mapping[str, AudioRecord], fingerprint: str,
        store: ArtifactStore, *, media_by_video: Mapping[str, Mapping[str, object]] | None = None,
        temporal: TemporalResolver | None = None,
    ) -> PipelineResult:
        """Normalize adapter output and atomically persist each independent modality."""
        frame_list = list(frames)
        if any(frame.preprocess_run_id != store.identity.preprocess_run_id for frame in frame_list):
            raise ValueError("TV1 frame preprocess_run_id does not match artifact store")
        if any(audio.preprocess_run_id != store.identity.preprocess_run_id for audio in audio_by_video.values()):
            raise ValueError("TV1 audio preprocess_run_id does not match artifact store")
        grouped: dict[str, list[FrameRecord]] = defaultdict(list)
        for frame in frame_list:
            grouped[frame.video_id].append(frame)
        resolver = temporal or LocalTemporalResolver(frame_list, [])
        statuses: list[ModalityStatus] = []
        video_ids = sorted(grouped)
        started_at = time.monotonic()
        for video_index, video_id in enumerate(video_ids, start=1):
            video_frames = sorted(grouped[video_id], key=lambda item: (item.timestamp_ms, item.frame_id))
            completed = self._completed_video_statuses(store, video_id, fingerprint)
            if completed is not None:
                statuses.extend(completed)
                elapsed = time.monotonic() - started_at
                print(
                    f"[WP04] {video_index}/{len(video_ids)} {video_id}: resumed "
                    f"(elapsed={elapsed:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            ocr_frames = self._sample_frames(video_frames, self._ocr_sample_interval_ms)
            object_frames = self._sample_frames(video_frames, self._object_sample_interval_ms)
            print(
                f"[WP04] {video_index}/{len(video_ids)} {video_id}: start "
                f"(frames={len(video_frames)}, ocr={len(ocr_frames)}, object={len(object_frames)})",
                file=sys.stderr,
                flush=True,
            )
            ocr_rows: list[OCRDetection] = []
            if not self._enabled["ocr"]:
                store.append_status(ModalityStatus(video_id, "ocr", fingerprint, "disabled"))
            else:
                try:
                    name, version = self._model(self._ocr, "ocr")
                    for batch in self._batches(ocr_frames, self._ocr_batch_size):
                        raw_by_frame = self._detect_batch(self._ocr, batch, OCRRawDetection, "ocr")
                        for frame, raw_detections in zip(batch, raw_by_frame):
                            ocr_rows.extend(
                                OCRDetection(frame.preprocess_run_id, video_id, frame.frame_id, frame.timestamp_ms,
                                             raw.text, normalize_text(raw.text).canonical, raw.bbox_xyxy_norm,
                                             raw.confidence, name, version, "pending")
                                for raw in raw_detections
                            )
                    store.commit_video("ocr", video_id, assign_ocr_evidence_ids(ocr_rows), fingerprint)
                except Exception as error:
                    store.append_status(ModalityStatus.failed(video_id, "ocr", fingerprint, str(error)))
            statuses.append(self._stored_status(store, video_id, "ocr"))

            asr_rows: list[ASRSegment] = []
            audio = audio_by_video.get(video_id)
            if not self._enabled["asr"]:
                store.append_status(ModalityStatus(video_id, "asr", fingerprint, "disabled"))
            elif audio is None:
                store.append_status(ModalityStatus.failed(video_id, "asr", fingerprint, "expected audio record missing"))
            elif not audio.declared_present:
                store.append_status(ModalityStatus.no_audio(video_id, "asr", fingerprint))
            else:
                try:
                    name, version = self._model(self._asr, "asr")
                    raw_segments = self._require_raw(self._asr.transcribe(audio), ASRRawSegment, "asr")
                    raw_segments.sort(key=lambda item: (item.start_ms, item.end_ms, item.text))
                    asr_rows = [
                        ASRSegment(audio.preprocess_run_id, video_id, f"asr:{video_id}:{index}", raw.start_ms, raw.end_ms,
                                   raw.text, normalize_text(raw.text).canonical, raw.confidence, name, version)
                        for index, raw in enumerate(raw_segments)
                    ]
                    store.commit_video("asr", video_id, asr_rows, fingerprint)
                except Exception as error:
                    store.append_status(ModalityStatus.failed(video_id, "asr", fingerprint, str(error)))
            statuses.append(self._stored_status(store, video_id, "asr"))

            if not self._enabled["object"]:
                store.append_status(ModalityStatus(video_id, "object", fingerprint, "disabled"))
            else:
                try:
                    name, version = self._model(self._objects, "object")
                    objects = []
                    for batch in self._batches(object_frames, self._object_batch_size):
                        raw_by_frame = self._detect_batch(self._objects, batch, ObjectRawDetection, "object")
                        objects.extend(
                            (frame, raw)
                            for frame, raw_detections in zip(batch, raw_by_frame)
                            for raw in raw_detections
                        )
                    objects.sort(key=lambda pair: (pair[0].frame_id, pair[1].bbox_xyxy_norm, pair[1].label))
                    object_rows = [
                        ObjectDetection(frame.preprocess_run_id, video_id, frame.frame_id, frame.timestamp_ms, raw.label,
                                        raw.bbox_xyxy_norm, raw.confidence, name, version,
                                        f"object:{video_id}:{frame.frame_id}:{index}")
                        for index, (frame, raw) in enumerate(objects)
                    ]
                    store.commit_video("object", video_id, object_rows, fingerprint)
                except Exception as error:
                    store.append_status(ModalityStatus.failed(video_id, "object", fingerprint, str(error)))
            statuses.append(self._stored_status(store, video_id, "object"))

            if not self._enabled["metadata"]:
                store.append_status(ModalityStatus(video_id, "metadata", fingerprint, "disabled"))
            else:
                try:
                    metadata = build_metadata(
                        store.identity.preprocess_run_id, video_id, video_frames,
                        (media_by_video or {}).get(video_id, {}),
                        assign_ocr_evidence_ids(ocr_rows),
                        asr_rows, resolver,
                    )
                    store.commit_video("metadata", video_id, [metadata], fingerprint)
                except Exception as error:
                    store.append_status(ModalityStatus.failed(video_id, "metadata", fingerprint, str(error)))
            statuses.append(self._stored_status(store, video_id, "metadata"))
            elapsed = time.monotonic() - started_at
            average = elapsed / video_index
            eta = average * (len(video_ids) - video_index)
            states = ",".join(status.state for status in statuses[-4:])
            print(
                f"[WP04] {video_index}/{len(video_ids)} {video_id}: done "
                f"(states={states}, elapsed={elapsed:.1f}s, eta={eta:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
        return PipelineResult(tuple(statuses))
