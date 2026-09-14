"""Bounded dense OCR over WP09-certified original frames."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .qa_query_plan import QaQueryPlan


_NUMBER = re.compile(r"(?<![\w])[-+]?\d{1,6}(?:[.,]\d{1,4})?(?![\w])")
_TIME_OR_DATE = re.compile(r"\b\d{1,4}[:/]\d{1,2}(?:[:/]\d{1,4})?\b")


@dataclass(frozen=True)
class ProvenJpegFrame:
    video_id: str
    frame_id: int
    timestamp_ms: int
    jpeg: bytes


@dataclass(frozen=True)
class DenseOcrReading:
    text: str
    normalized_value: str
    frame_id: int
    timestamp_ms: int
    confidence: float
    bbox_xyxy_norm: tuple[float, float, float, float]


@dataclass(frozen=True)
class DenseOcrResult:
    available: bool
    selected: DenseOcrReading | None = None
    support_frame_ids: tuple[int, ...] = ()
    inspected_frame_count: int = 0
    reason: str | None = None

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def _transport(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    request = Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310 -- configured local OCR service
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("dense OCR service unavailable") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("dense OCR response must be an object")
    return value


def _normalize_number(value: str) -> str:
    clean = value.strip().replace(",", ".")
    if clean.startswith("+"):
        clean = clean[1:]
    return clean


class DenseOcrReader:
    """Read transient values only after exact-frame proof has been supplied."""

    def __init__(
        self, *, enabled: bool, base_url: str = "http://127.0.0.1:8765",
        timeout_seconds: float = 120.0, max_frames: int = 64,
        max_batch_frames: int = 4,
        min_confidence: float = 0.45, min_stable_frames: int = 2,
        transport: Transport | None = None,
    ) -> None:
        if (
            timeout_seconds <= 0 or not 1 <= max_frames <= 128
            or not 1 <= max_batch_frames <= max_frames
            or not 1 <= min_stable_frames <= max_frames
        ):
            raise ValueError("dense OCR limits are invalid")
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_frames = max_frames
        self.max_batch_frames = max_batch_frames
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.min_stable_frames = min_stable_frames
        self._transport = transport or _transport

    @classmethod
    def from_environment(cls) -> "DenseOcrReader":
        return cls(
            enabled=os.environ.get("TV4_DENSE_QA_ENABLED", "1") == "1",
            base_url=os.environ.get("TV4_DENSE_OCR_BASE_URL", "http://127.0.0.1:8765"),
            timeout_seconds=float(os.environ.get("TV4_DENSE_OCR_TIMEOUT_SECONDS", "120")),
            max_frames=int(os.environ.get("TV4_DENSE_QA_MAX_FRAMES", "64")),
            max_batch_frames=int(os.environ.get("TV4_DENSE_OCR_MAX_BATCH_FRAMES", "4")),
            min_confidence=float(os.environ.get("TV4_DENSE_OCR_MIN_CONFIDENCE", "0.45")),
            min_stable_frames=int(os.environ.get("TV4_DENSE_OCR_MIN_STABLE_FRAMES", "2")),
        )

    def read(self, frames: Sequence[ProvenJpegFrame], plan: QaQueryPlan) -> DenseOcrResult:
        if not self.enabled:
            return DenseOcrResult(False, reason="disabled")
        if not plan.requires_dense_read or plan.answer_source != "device_display":
            return DenseOcrResult(False, reason="question_not_dense_readable")
        selected_frames = tuple(frames[: self.max_frames])
        if not selected_frames:
            return DenseOcrResult(False, reason="certified_frames_unavailable")
        try:
            with tempfile.TemporaryDirectory(prefix="tv4-dense-ocr-") as temp_dir:
                paths: list[str] = []
                for index, frame in enumerate(selected_frames):
                    path = Path(temp_dir) / f"{index:03d}-{frame.frame_id}.jpg"
                    path.write_bytes(frame.jpeg)
                    paths.append(str(path))
                rows_by_frame: list[object] = []
                for start in range(0, len(paths), self.max_batch_frames):
                    chunk = paths[start : start + self.max_batch_frames]
                    try:
                        payload = self._transport(
                            f"{self.base_url}/ocr/batch", {"keyframe_paths": chunk}, self.timeout_seconds,
                        )
                        chunk_rows = payload.get("detections")
                        if not isinstance(chunk_rows, list) or len(chunk_rows) != len(chunk):
                            raise RuntimeError("malformed dense OCR batch")
                        rows_by_frame.extend(chunk_rows)
                    except Exception:
                        # Some CRAFT deployments cannot batch particular image
                        # shapes. Preserve bounded progress with single-image
                        # inference instead of discarding the whole shot.
                        for path in chunk:
                            payload = self._transport(
                                f"{self.base_url}/ocr", {"keyframe_path": path}, self.timeout_seconds,
                            )
                            rows_by_frame.append(payload.get("detections", []))
        except Exception as exc:
            return DenseOcrResult(
                False, inspected_frame_count=len(selected_frames), reason=type(exc).__name__,
            )
        if not isinstance(rows_by_frame, list) or len(rows_by_frame) != len(selected_frames):
            return DenseOcrResult(False, inspected_frame_count=len(selected_frames), reason="malformed_ocr_response")
        readings: list[tuple[int, DenseOcrReading]] = []
        for index, (frame, rows) in enumerate(zip(selected_frames, rows_by_frame)):
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                text = row.get("text")
                confidence = row.get("confidence")
                bbox = row.get("bbox_xyxy_norm")
                if (
                    not isinstance(text, str)
                    or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or float(confidence) < self.min_confidence
                    or not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                ):
                    continue
                # Broadcast clocks/dates are stable overlays and otherwise
                # beat the actual transient device reading. They remain valid
                # only when the explicitly requested device is a timer.
                if plan.device != "timer" and _TIME_OR_DATE.search(text):
                    continue
                for match in _NUMBER.finditer(text):
                    readings.append((index, DenseOcrReading(
                        text=text,
                        normalized_value=_normalize_number(match.group(0)),
                        frame_id=frame.frame_id,
                        timestamp_ms=frame.timestamp_ms,
                        confidence=float(confidence),
                        bbox_xyxy_norm=tuple(float(value) for value in bbox),
                    )))
        choice = self._select(readings, plan)
        if choice is None:
            return DenseOcrResult(False, inspected_frame_count=len(selected_frames), reason="no_stable_numeric_reading")
        selected, support = choice
        return DenseOcrResult(
            True, selected=selected,
            support_frame_ids=tuple(readings[item][1].frame_id for item in support),
            inspected_frame_count=len(selected_frames),
        )

    def health(self, timeout: float = 2.0) -> dict[str, object]:
        url = f"{self.base_url}/health"
        if not self.enabled:
            return {"available": False, "enabled": False, "reason": "disabled", "url": url}
        try:
            with urlopen(url, timeout=timeout) as response:  # nosec B310 -- configured local OCR service
                payload = json.loads(response.read().decode("utf-8"))
            return {"available": True, "enabled": True, "url": url, "response": payload}
        except Exception as exc:
            return {"available": False, "enabled": True, "url": url, "reason": type(exc).__name__}

    def _select(
        self, readings: Sequence[tuple[int, DenseOcrReading]], plan: QaQueryPlan,
    ) -> tuple[DenseOcrReading, tuple[int, ...]] | None:
        by_value: dict[str, list[int]] = {}
        for reading_index, (_, reading) in enumerate(readings):
            by_value.setdefault(reading.normalized_value, []).append(reading_index)
        stable_runs: list[tuple[int, float, tuple[int, ...]]] = []
        for indexes in by_value.values():
            run: list[int] = []
            previous_position: int | None = None
            for reading_index in indexes:
                position = readings[reading_index][0]
                if previous_position is not None and position - previous_position > 2:
                    if len({readings[item][0] for item in run}) >= self.min_stable_frames:
                        stable_runs.append(self._run_score(readings, run))
                    run = []
                run.append(reading_index)
                previous_position = position
            if len({readings[item][0] for item in run}) >= self.min_stable_frames:
                stable_runs.append(self._run_score(readings, run))
        if not stable_runs:
            return None
        if plan.temporal_rule == "last_before_scene_change":
            stable_runs.sort(key=lambda item: (-item[0], -item[1]))
        else:
            stable_runs.sort(key=lambda item: (-len(item[2]), -item[1], -item[0]))
        _, _, support = stable_runs[0]
        ordered = sorted(support, key=lambda item: readings[item][0])
        # A middle frame is less likely to be motion-blurred or already inside
        # the next shot than the final occurrence of the same stable reading.
        selected_index = ordered[len(ordered) // 2]
        return readings[selected_index][1], support

    @staticmethod
    def _run_score(
        readings: Sequence[tuple[int, DenseOcrReading]], indexes: Sequence[int],
    ) -> tuple[int, float, tuple[int, ...]]:
        return (
            max(readings[index][0] for index in indexes),
            sum(readings[index][1].confidence for index in indexes) / len(indexes),
            tuple(indexes),
        )
