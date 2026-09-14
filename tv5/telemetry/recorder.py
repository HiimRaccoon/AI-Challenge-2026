"""Thread-safe, bounded operational telemetry recorder."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from threading import Lock
from typing import Iterator

from .events import TelemetryEvent


class TelemetryRecorder:
    def __init__(self, max_events: int = 10000, output_path: Path | None = None) -> None:
        self._max_events = max_events
        self._output_path = output_path
        self._events: deque[TelemetryEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def record(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._events.append(event)
            if self._output_path:
                try:
                    self._output_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._output_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(event.to_dict()) + "\n")
                except OSError:
                    # Non-blocking telemetry output failure
                    pass

    def get_events(self) -> list[TelemetryEvent]:
        with self._lock:
            return list(self._events)

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def iter_events(self) -> Iterator[TelemetryEvent]:
        with self._lock:
            for ev in list(self._events):
                yield ev
