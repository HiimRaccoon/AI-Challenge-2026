"""Thin client for TV1's WP06 API server (tv1/wp06_api_server.py).

TV4 never re-derives frame_id/timestamp itself: any frame/window/neighbor
information always comes from this service so TV4's output stays consistent
with the "video gốc là sự thật cuối cùng" rule TV1 documents.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any


class TV1ClientError(RuntimeError):
    pass


class TV1Client:
    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise TV1ClientError(f"GET {url} failed: {exc}") from exc

    def health(self) -> dict:
        return self._get("/health")

    def summary(self) -> dict:
        return self._get("/summary")

    def manifest(self) -> Any:
        return self._get("/manifest")

    def media(self) -> list[dict]:
        """List of media probe metadata for all videos."""
        return self._get("/media")

    def frames(self, video_id: str) -> list[dict]:
        """List of {frame_id, timestamp_ms, keyframe_path, ...} for a video."""
        return self._get(f"/frames/{video_id}")

    def keyframe_image_url(self, video_id: str, filename: str) -> str:
        return f"{self.base_url}/keyframe-image/{video_id}/{filename}"

    def validate_run(self, run_id: str) -> dict:
        return self._get(f"/runs/{run_id}/validate")

    def nearest_frame(self, video_id: str, timestamp_ms: int) -> dict | None:
        """Best-effort: pick the frame whose timestamp is closest to the target.

        Used by WP10/WP12 when a modality only returns a timestamp window
        (e.g. an ASR segment) and needs a concrete representative frame_id.
        """
        frames = self.frames(video_id)
        if not frames:
            return None
        return min(frames, key=lambda f: abs(int(f["timestamp_ms"]) - int(timestamp_ms)))

    def frames_in_window(self, video_id: str, start_ms: int, end_ms: int) -> list[dict]:
        return [f for f in self.frames(video_id) if start_ms <= int(f["timestamp_ms"]) <= end_ms]
