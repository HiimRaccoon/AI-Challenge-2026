"""Adapters for approved local model commands without embedding vendor model source."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlsplit

from .adapters import ASRRawSegment, AdapterUnavailableError, OCRRawDetection, ObjectRawDetection
from .contracts import AudioRecord, FrameRecord


@dataclass(slots=True)
class _CommandAdapter:
    command_env: str
    model_name: str
    model_version: str = "local"
    vad_version: str = "local"
    server_url: str | None = None

    # A preprocess run sends many sequential requests to one localhost server.
    # Keep one HTTP/1.1 connection rather than opening a TCP connection per batch.
    _server_connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = field(
        default=None, init=False, repr=False,
    )
    _server_origin: tuple[str, str, int] | None = field(default=None, init=False, repr=False)

    def _close_server_connection(self) -> None:
        if self._server_connection is not None:
            self._server_connection.close()
        self._server_connection = None
        self._server_origin = None

    def _connection_for_server(self) -> tuple[http.client.HTTPConnection | http.client.HTTPSConnection, str]:
        if not self.server_url:
            raise AdapterUnavailableError("model server URL is not configured")
        parsed = urlsplit(self.server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise AdapterUnavailableError(f"model server URL is invalid: {self.server_url}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        origin = (parsed.scheme, parsed.hostname, port)
        if self._server_connection is None or self._server_origin != origin:
            self._close_server_connection()
            connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            self._server_connection = connection_type(parsed.hostname, port, timeout=120)
            self._server_origin = origin
        return self._server_connection, parsed.path.rstrip("/")

    def _server_request(self, route: str, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(2):
            try:
                connection, base_path = self._connection_for_server()
                connection.request(
                    "POST", f"{base_path}/{route.lstrip('/')}", body=body,
                    headers={"Content-Type": "application/json", "Connection": "keep-alive"},
                )
                response = connection.getresponse()
                response_body = response.read().decode("utf-8", errors="replace")
                if response.status >= 400:
                    raise AdapterUnavailableError(
                        f"model server {route} failed ({response.status}): {response_body}"
                    )
                return json.loads(response_body)
            except AdapterUnavailableError:
                raise
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                self._close_server_connection()
                if attempt == 1:
                    raise AdapterUnavailableError(f"model server {route} is unavailable: {error}") from error
            except json.JSONDecodeError as error:
                raise AdapterUnavailableError(f"model server {route} returned invalid JSON: {error}") from error
        raise AssertionError("unreachable")

    def _run(self, **values: str) -> Any:
        command = os.environ.get(self.command_env, "").strip()
        if not command:
            raise AdapterUnavailableError(f"local model command is not configured: {self.command_env}")
        try:
            completed = subprocess.run(
                command.format(**values),
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return json.loads(completed.stdout)
        except subprocess.CalledProcessError as error:
            raise AdapterUnavailableError(f"local model command failed ({self.command_env}): {error.stderr.strip()}") from error
        except json.JSONDecodeError as error:
            raise AdapterUnavailableError(f"local model command must emit JSON ({self.command_env})") from error


class CommandOCRAdapter(_CommandAdapter):
    def detect(self, frame: FrameRecord) -> list[OCRRawDetection]:
        if not frame.keyframe_path:
            raise AdapterUnavailableError("OCR requires FrameRecord.keyframe_path")
        output = (
            self._server_request("ocr", {"keyframe_path": frame.keyframe_path})
            if self.server_url
            else self._run(keyframe_path=frame.keyframe_path)
        )
        rows = output.get("detections", output) if isinstance(output, dict) else output
        if not isinstance(rows, list):
            raise AdapterUnavailableError("OCR command JSON must be a list or {detections: [...]}")
        return [OCRRawDetection(str(row["text"]), tuple(row["bbox_xyxy_norm"]), float(row["confidence"])) for row in rows]

    def detect_batch(self, frames: Sequence[FrameRecord]) -> list[list[OCRRawDetection]]:
        if not frames:
            return []
        if not self.server_url:
            return [self.detect(frame) for frame in frames]
        paths = [frame.keyframe_path for frame in frames]
        if any(path is None for path in paths):
            raise AdapterUnavailableError("OCR requires FrameRecord.keyframe_path")
        output = self._server_request("ocr/batch", {"keyframe_paths": paths})
        rows_by_frame = output.get("detections", output) if isinstance(output, dict) else output
        if not isinstance(rows_by_frame, list) or len(rows_by_frame) != len(frames):
            raise AdapterUnavailableError("OCR batch response must contain one detection list per keyframe")
        results: list[list[OCRRawDetection]] = []
        for rows in rows_by_frame:
            if not isinstance(rows, list):
                raise AdapterUnavailableError("OCR batch response contains a non-list detection result")
            results.append([
                OCRRawDetection(str(row["text"]), tuple(row["bbox_xyxy_norm"]), float(row["confidence"]))
                for row in rows
            ])
        return results


@dataclass(slots=True)
class CommandASRAdapter(_CommandAdapter):
    vad_enabled: bool | None = None
    vad_threshold: float | None = None
    min_speech_ms: int | None = None
    max_segment_ms: int | None = None
    overlap_ms: int | None = None

    def transcribe(self, audio: AudioRecord) -> list[ASRRawSegment]:
        if not audio.audio_path:
            raise AdapterUnavailableError("ASR requires AudioRecord.audio_path")
        payload: dict[str, Any] = {"audio_path": audio.audio_path}
        for key in ("vad_enabled", "vad_threshold", "min_speech_ms", "max_segment_ms", "overlap_ms"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        output = (
            self._server_request("asr", payload)
            if self.server_url
            else self._run(audio_path=audio.audio_path)
        )
        rows = output.get("segments", output) if isinstance(output, dict) else output
        if not isinstance(rows, list):
            raise AdapterUnavailableError("ASR command JSON must be a list or {segments: [...]}")
        return [ASRRawSegment(int(row["start_ms"]), int(row["end_ms"]), str(row["text"]), float(row["confidence"])) for row in rows]


class CommandObjectAdapter(_CommandAdapter):
    def detect(self, frame: FrameRecord) -> list[ObjectRawDetection]:
        if not frame.keyframe_path:
            raise AdapterUnavailableError("object detection requires FrameRecord.keyframe_path")
        output = (
            self._server_request("object", {"keyframe_path": frame.keyframe_path})
            if self.server_url
            else self._run(keyframe_path=frame.keyframe_path)
        )
        rows = output.get("detections", output) if isinstance(output, dict) else output
        if not isinstance(rows, list):
            raise AdapterUnavailableError("object command JSON must be a list or {detections: [...]}")
        return [ObjectRawDetection(str(row["label"]), tuple(row["bbox_xyxy_norm"]), float(row["confidence"])) for row in rows]

    def detect_batch(self, frames: Sequence[FrameRecord]) -> list[list[ObjectRawDetection]]:
        if not frames:
            return []
        if not self.server_url:
            return [self.detect(frame) for frame in frames]
        paths = [frame.keyframe_path for frame in frames]
        if any(path is None for path in paths):
            raise AdapterUnavailableError("object detection requires FrameRecord.keyframe_path")
        output = self._server_request("object/batch", {"keyframe_paths": paths})
        rows_by_frame = output.get("detections", output) if isinstance(output, dict) else output
        if not isinstance(rows_by_frame, list) or len(rows_by_frame) != len(frames):
            raise AdapterUnavailableError("object batch response must contain one detection list per keyframe")
        results: list[list[ObjectRawDetection]] = []
        for rows in rows_by_frame:
            if not isinstance(rows, list):
                raise AdapterUnavailableError("object batch response contains a non-list detection result")
            results.append([
                ObjectRawDetection(str(row["label"]), tuple(row["bbox_xyxy_norm"]), float(row["confidence"]))
                for row in rows
            ])
        return results


def build_ocr_adapter(
    command_env: str = "WP04_OCR_COMMAND", model_version: str = "local", server_url: str | None = None,
) -> CommandOCRAdapter:
    return CommandOCRAdapter(command_env, "craft-parseq-vn", model_version, server_url=server_url)


def build_asr_adapter(
    command_env: str = "WP04_CHUNKFORMER_COMMAND", model_version: str = "local", vad_version: str = "local",
    server_url: str | None = None, vad_enabled: bool | None = None, vad_threshold: float | None = None,
    min_speech_ms: int | None = None, max_segment_ms: int | None = None, overlap_ms: int | None = None,
) -> CommandASRAdapter:
    return CommandASRAdapter(
        command_env=command_env, model_name="chunkformer-ctc-large-vie", model_version=model_version,
        vad_version=vad_version, server_url=server_url, vad_enabled=vad_enabled,
        vad_threshold=vad_threshold, min_speech_ms=min_speech_ms,
        max_segment_ms=max_segment_ms, overlap_ms=overlap_ms,
    )


def build_object_adapter(
    command_env: str = "WP04_RFDETR_COMMAND", model_version: str = "local", server_url: str | None = None,
) -> CommandObjectAdapter:
    return CommandObjectAdapter(command_env, "rf-detr", model_version, server_url=server_url)
