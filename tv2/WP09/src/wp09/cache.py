"""Internal cache for frames already resolved by a canonical mapping run."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from time import monotonic
from typing import Callable

from .decoder import DecodedFrame


@dataclass(frozen=True)
class DecodedWindowKey:
    preprocess_run_id: str
    video_id: str
    start_pts: int
    end_pts: int
    decoder_config: str


@dataclass(frozen=True)
class WindowRequestKey:
    """Lookup identity available before any decode. Values are not canonical IDs."""
    preprocess_run_id: str
    video_id: str
    center_ms: int
    radius_ms: int
    budget_signature: tuple[int, int, int | None, int]
    decoder_config: str


@dataclass(frozen=True)
class CachedWindow:
    key: DecodedWindowKey
    frames: tuple[DecodedFrame, ...]
    window_start_ms: int
    window_end_ms: int
    best_frame_id: int
    budget_exhausted: bool


@dataclass(frozen=True)
class _TimedFrames:
    frames: tuple[DecodedFrame, ...]
    expires_at: float


class DecodedWindowCache:
    """Bounded in-memory cache whose canonical identity includes mapping/run truth."""

    def __init__(
        self,
        max_entries: int = 32,
        ttl_seconds: float = 300.0,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive number")
        self._max_entries = max_entries
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[DecodedWindowKey, _TimedFrames] = OrderedDict()
        self._request_index: OrderedDict[WindowRequestKey, CachedWindow] = OrderedDict()

    def get(self, key: DecodedWindowKey) -> tuple[DecodedFrame, ...] | None:
        self._purge_expired()
        return self._read(key)

    def put(self, key: DecodedWindowKey, frames: tuple[DecodedFrame, ...]) -> None:
        self._purge_expired()
        self._drop_request_aliases(key)
        self._store(key, frames)

    def get_for_request(self, key: WindowRequestKey) -> CachedWindow | None:
        self._purge_expired()
        entry = self._request_index.get(key)
        if entry is None:
            return None
        self._request_index.move_to_end(key)
        frames = self._read(entry.key)
        if frames is None:
            self._request_index.pop(key, None)
            return None
        return replace(entry, frames=frames)

    def put_for_request(self, request_key: WindowRequestKey, entry: CachedWindow) -> None:
        self._purge_expired()
        self._store(entry.key, entry.frames)
        self._request_index[request_key] = entry
        self._request_index.move_to_end(request_key)
        while len(self._request_index) > self._max_entries:
            self._request_index.popitem(last=False)

    def _read(self, key: DecodedWindowKey) -> tuple[DecodedFrame, ...] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries[key] = _TimedFrames(entry.frames, self._clock() + self._ttl_seconds)
        self._entries.move_to_end(key)
        return entry.frames

    def _store(self, key: DecodedWindowKey, frames: tuple[DecodedFrame, ...]) -> None:
        self._entries[key] = _TimedFrames(frames, self._clock() + self._ttl_seconds)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            evicted_key, _ = self._entries.popitem(last=False)
            self._drop_request_aliases(evicted_key)

    def _purge_expired(self) -> None:
        now = self._clock()
        for key, entry in tuple(self._entries.items()):
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                self._drop_request_aliases(key)

    def _drop_request_aliases(self, decoded_key: DecodedWindowKey) -> None:
        for request_key, entry in tuple(self._request_index.items()):
            if entry.key == decoded_key:
                self._request_index.pop(request_key, None)
