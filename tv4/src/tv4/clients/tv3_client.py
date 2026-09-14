"""Client for TV3's FastAPI backend (tv1tv3/TV1_TV3_WP04/src/aic2026/api.py).

Talks the documented `SearchRequest` -> list[`SearchCandidate`] contract over
HTTP (POST /text/search, /ocr/search, /asr/search, /object/search,
/metadata/search). TV4 never reads TV3's on-disk artifacts directly.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from ..contracts import SearchCandidate, SearchRequest

_ROUTE_BY_SOURCE = {
    "text": "/text/search",
    "ocr": "/ocr/search",
    "asr": "/asr/search",
    "object": "/object/search",
    "metadata": "/metadata/search",
}


class TV3ClientError(RuntimeError):
    pass


class TV3Client:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_payload(self, path: str, payload: dict) -> object:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Callers that need a truthful degraded state (notably VQA) must
            # distinguish an unavailable artifact from an empty result set.
            raise TV3ClientError(f"POST {url} failed ({exc.code}): {exc.read()[:300]}") from exc
        except urllib.error.URLError as exc:
            raise TV3ClientError(f"POST {url} failed: {exc}") from exc
        except TimeoutError as exc:
            # A timeout while reading the response (http.client.getresponse)
            # surfaces as a bare TimeoutError, not wrapped in URLError -- must
            # be caught separately or it crashes the whole batch instead of
            # failing just this one query/route.
            raise TV3ClientError(f"POST {url} timed out after {self.timeout}s") from exc

    def _post(self, path: str, payload: dict) -> list[dict]:
        response = self._post_payload(path, payload)
        if not isinstance(response, list) or not all(isinstance(row, dict) for row in response):
            raise TV3ClientError(f"POST {path} returned malformed candidate rows")
        return response

    def health(self, timeout: float = 2.0) -> dict[str, object]:
        url = f"{self.base_url}/health"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return {"available": True, "url": url, "response": payload}
        except Exception as exc:
            return {"available": False, "url": url, "reason": type(exc).__name__}

    def search(self, route: str, request: SearchRequest) -> list[SearchCandidate]:
        if route not in _ROUTE_BY_SOURCE:
            raise ValueError(f"unknown TV3 route: {route}")
        rows = self._post(_ROUTE_BY_SOURCE[route], request.to_json())
        return [SearchCandidate.from_json(r) for r in rows]

    def _get_payload(self, path: str, params: dict[str, object]) -> object:
        query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise TV3ClientError(f"GET {url} failed ({exc.code}): {exc.read()[:300]}") from exc
        except urllib.error.URLError as exc:
            raise TV3ClientError(f"GET {url} failed: {exc}") from exc
        except TimeoutError as exc:
            raise TV3ClientError(f"GET {url} timed out after {self.timeout}s") from exc

    def _get_records(self, path: str, params: dict[str, object]) -> list[dict]:
        payload = self._get_payload(path, params)
        if isinstance(payload, dict):
            payload = payload.get("items", [])
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise TV3ClientError(f"GET {path} returned malformed evidence records")
        return payload

    # These are deliberately thin public-WP04 catalogue adapters. TV4 retains
    # the returned producer records rather than rebuilding evidence itself.
    def get_ocr_detections(self, *, video_id: str, frame_id: int) -> list[dict]:
        return self._get_records("/ocr/detections", {"video_id": video_id, "frame_id": frame_id, "limit": 100})

    def get_asr_segments(self, *, video_id: str, start_ms: int, end_ms: int) -> list[dict]:
        return self._get_records("/asr/segments", {"video_id": video_id, "start_ms": start_ms, "end_ms": end_ms, "limit": 100})

    def get_asr_context(self, *, segment_id: str) -> list[dict]:
        payload = self._get_payload(f"/asr/{urllib.parse.quote(segment_id, safe='')}/context", {})
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise TV3ClientError("ASR context returned malformed evidence records")
        return [row for row in payload["segments"] if isinstance(row, dict)]

    def search_transcript_windows(
        self, *, query_id: str, query_text: str, limit: int = 20, context_ms: int = 10_000,
    ) -> list[dict]:
        """Find spoken text and return TV1-canonical frames spanning each hit.

        The response is intentionally kept as producer-owned records rather
        than converted to ``SearchCandidate``: one transcript match maps to a
        time interval and many valid frames, not one synthetic midpoint.
        """
        if not query_id or not query_text.strip() or not 1 <= limit <= 100 or not 0 <= context_ms <= 30_000:
            raise ValueError("invalid transcript-window search request")
        payload = self._post_payload(
            "/asr/windows/search",
            {
                "query_id": query_id, "query_text": query_text, "limit": limit,
                "context_ms": context_ms, "mode": "semantic",
            },
        )
        if not isinstance(payload, dict) or payload.get("query_id") != query_id:
            raise TV3ClientError("transcript window search returned malformed payload")
        windows = payload.get("windows")
        if not isinstance(windows, list) or not all(isinstance(item, dict) for item in windows):
            raise TV3ClientError("transcript window search returned malformed windows")
        valid: list[dict] = []
        for window in windows:
            start_ms, end_ms = window.get("start_ms"), window.get("end_ms")
            if (
                not isinstance(window.get("video_id"), str)
                or isinstance(start_ms, bool) or not isinstance(start_ms, int)
                or isinstance(end_ms, bool) or not isinstance(end_ms, int)
                or start_ms < 0 or end_ms < start_ms
            ):
                raise TV3ClientError("transcript window has invalid temporal identity")
            frames = window.get("canonical_frames", [])
            if not isinstance(frames, list) or not all(
                isinstance(frame, dict)
                and isinstance(frame.get("frame_id"), int) and not isinstance(frame.get("frame_id"), bool)
                and isinstance(frame.get("timestamp_ms"), int) and not isinstance(frame.get("timestamp_ms"), bool)
                for frame in frames
            ):
                raise TV3ClientError("transcript window has malformed canonical frames")
            valid.append(window)
        return valid

    def get_object_detections(self, *, video_id: str, frame_id: int) -> list[dict]:
        return self._get_records("/objects/detections", {"video_id": video_id, "frame_id": frame_id, "limit": 100})

    def get_metadata_records(self, *, video_id: str) -> list[dict]:
        return self._get_records("/metadata/records", {"video_id": video_id, "limit": 100})

    def get_evidence_window(
        self, *, video_id: str, timestamp_ms: int, window_ms: int, limit: int = 100,
    ) -> dict[str, object]:
        """Read bounded, video-local text evidence without another corpus search.

        This is used for factual VQA questions whose answer appears shortly
        before or after the retrieval anchor. The producer owns the temporal
        lookup and returns its own OCR/ASR text; TV4 never derives frame IDs.
        """
        if timestamp_ms < 0 or not 0 <= window_ms <= 30_000 or not 1 <= limit <= 100:
            raise ValueError("invalid evidence window request")
        url = f"{self.base_url}/evidence/window"
        body = json.dumps(
            {
                "video_id": video_id,
                "timestamp_ms": timestamp_ms,
                "window_ms": window_ms,
                "limit": limit,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise TV3ClientError(f"POST {url} failed ({exc.code}): {exc.read()[:300]}") from exc
        except urllib.error.URLError as exc:
            raise TV3ClientError(f"POST {url} failed: {exc}") from exc
        except TimeoutError as exc:
            raise TV3ClientError(f"POST {url} timed out after {self.timeout}s") from exc
        if not isinstance(payload, dict):
            raise TV3ClientError("evidence window returned malformed payload")
        result: dict[str, object] = {}
        for branch in ("ocr_texts", "asr_texts", "object_labels"):
            values = payload.get(branch, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise TV3ClientError("evidence window returned malformed text values")
            result[branch] = [value for value in values if value.strip()]
        ocr_records = payload.get("ocr_records", [])
        if not isinstance(ocr_records, list) or not all(isinstance(record, dict) for record in ocr_records):
            raise TV3ClientError("evidence window returned malformed OCR records")
        result["ocr_records"] = ocr_records
        return result

    def search_all(self, request: SearchRequest, routes: Iterable[str]) -> dict[str, list[SearchCandidate]]:
        """Search independent TV3 indexes concurrently.

        Each route has its own HTTP request and may take up to the client
        timeout. Running them serially made a VQA request wait for every
        unavailable index in turn. Preserve the caller's route order in the
        result so fusion remains deterministic while bounding latency by the
        slowest branch instead of their sum.
        """
        ordered_routes = tuple(dict.fromkeys(routes))
        if not ordered_routes:
            return {}

        def _search_or_empty(route: str) -> list[SearchCandidate]:
            try:
                return self.search(route, request)
            except TV3ClientError:
                return []

        with ThreadPoolExecutor(max_workers=len(ordered_routes), thread_name_prefix="tv3-search") as executor:
            futures = {route: executor.submit(_search_or_empty, route) for route in ordered_routes}
            return {route: futures[route].result() for route in ordered_routes}
