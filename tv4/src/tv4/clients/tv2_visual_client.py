"""Subprocess client for TV2's WP03 visual retrieval CLI.

WP03 ships as a CLI (`python -m wp03 search ...`) that must run inside its
own GPU-enabled virtualenv (see WP03/envs/*.txt and configs/runtime.*.yaml).
TV4 therefore shells out to that interpreter instead of importing wp03
in-process, so TV4 itself keeps working on machines without the four visual
model stacks installed.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import subprocess
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..contracts import SearchCandidate


class TV2VisualClientError(RuntimeError):
    pass


@dataclass
class TV2VisualClient:
    python_executable: str  # path to WP03's venv python, e.g. WP03/.venv/Scripts/python.exe
    wp03_cwd: Path          # WP03 repo root (so `-m wp03` resolves)
    artifact_root: Path
    runtime_root: Path | None = None
    runtime_profile: Path | None = None
    top_k: int = 100
    candidate_k_per_model: int = 200
    enabled: bool = True
    persistent: bool = False
    timeout_seconds: float = 20.0
    warmup_enabled: bool = False
    warmup_timeout_seconds: float = 180.0
    warmup_query: str = "a person in a lecture slide"
    warmup_models: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    temporal_models: tuple[str, ...] = ()
    multi_moment_models: tuple[str, ...] = ()
    _last_diagnostics: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _process_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _warmup_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _warmup_state: str = field(default="disabled", init=False, repr=False)
    _warmup_error: str | None = field(default=None, init=False, repr=False)
    _warmup_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.persistent:
            atexit.register(self.close)
        if self.warmup_enabled:
            self._warmup_state = "pending"

    def _server_command(self) -> list[str]:
        command = [
            self.python_executable, "-m", "wp03", "serve-stdio",
            "--artifact-root", str(self.artifact_root),
            "--top-k", str(self.top_k),
            "--candidate-k-per-model", str(self.candidate_k_per_model),
        ]
        if self.runtime_root:
            command += ["--runtime-root", str(self.runtime_root)]
        if self.runtime_profile:
            command += ["--runtime-profile", str(self.runtime_profile)]
        if self.models:
            command += ["--models", ",".join(self.models)]
        return command

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = subprocess.Popen(
            self._server_command(), cwd=str(self.wp03_cwd),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        return self._process

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

    def _persistent_request(
        self, payload: dict[str, object], *, timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        budget = self.timeout_seconds if timeout_seconds is None else max(0.01, timeout_seconds)
        started = time.monotonic()
        if not self._process_lock.acquire(timeout=budget):
            raise TV2VisualClientError(
                f"WP03 request queue exceeded {budget:g}s budget"
            )
        try:
            process = self._ensure_process()
            if process.stdin is None or process.stdout is None:
                raise TV2VisualClientError("WP03 stdio coordinator has no pipes")
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(process.stdout.readline)
            try:
                remaining = max(0.01, budget - (time.monotonic() - started))
                line = future.result(timeout=remaining)
            except FutureTimeoutError as exc:
                self.close()
                raise TV2VisualClientError(
                    f"WP03 search exceeded {budget:g}s budget"
                ) from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if not line:
                self.close()
                raise TV2VisualClientError("WP03 stdio coordinator exited")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TV2VisualClientError(f"WP03 returned invalid JSON: {line[:300]}") from exc
            if not isinstance(response, dict) or response.get("status") == "error":
                raise TV2VisualClientError(str(response.get("error", "WP03 request failed")))
            return response
        finally:
            self._process_lock.release()

    def start_warmup(self) -> None:
        """Start one background model warm-up without blocking API startup."""
        if not (self.enabled and self.persistent and self.warmup_enabled):
            return
        with self._warmup_lock:
            if self._warmup_state != "pending":
                return
            self._warmup_state = "running"
            self._warmup_error = None
            self._warmup_thread = threading.Thread(
                target=self._run_warmup,
                name="wp03-visual-warmup",
                daemon=True,
            )
            self._warmup_thread.start()

    def _run_warmup(self) -> None:
        try:
            requested_models = self.warmup_models or self.models
            payload = self._persistent_request(
                {
                    "operation": "search",
                    "query_id": "tv4-startup-warmup",
                    "query_text": self.warmup_query,
                    "event_index": None,
                    "top_k": 1,
                    "candidate_k_per_model": 1,
                    "models": ",".join(requested_models),
                },
                timeout_seconds=self.warmup_timeout_seconds,
            )
            self._record_diagnostics(payload)
            models_used = payload.get("models_used")
            if isinstance(models_used, list) and not models_used:
                raise TV2VisualClientError("WP03 warm-up completed without an available model")
            if isinstance(models_used, list):
                missing = sorted(set(requested_models) - set(models_used))
                if missing:
                    raise TV2VisualClientError(
                        f"WP03 warm-up did not load required models: {', '.join(missing)}"
                    )
        except Exception as exc:
            with self._warmup_lock:
                self._warmup_state = "failed"
                self._warmup_error = str(exc)
            return
        with self._warmup_lock:
            self._warmup_state = "ready"
            self._warmup_error = None

    def health(self) -> dict[str, object]:
        checks = {
            "python_executable": Path(self.python_executable).is_file(),
            "wp03_cwd": self.wp03_cwd.is_dir(),
            "artifact_root": self.artifact_root.is_dir(),
            "runtime_profile": self.runtime_profile.is_file() if self.runtime_profile else False,
        }
        configured = bool(self.enabled and all(checks.values()))
        models_used = self._last_diagnostics.get("models_used")
        runtime_available = not isinstance(models_used, list) or bool(models_used)
        degraded = bool(self._last_diagnostics.get("degraded", False))
        warmup_ready = not self.warmup_enabled or self._warmup_state == "ready"
        result: dict[str, object] = {
            "available": bool(configured and runtime_available and warmup_ready),
            "enabled": self.enabled,
            "checks": checks,
            "degraded": degraded,
            "warmup": {
                "enabled": self.warmup_enabled,
                "status": self._warmup_state,
                "error": self._warmup_error,
            },
        }
        if self._last_diagnostics:
            result["last_search"] = dict(self._last_diagnostics)
        return result

    def _record_diagnostics(self, payload: dict[str, object]) -> None:
        self._last_diagnostics = {
            key: payload[key]
            for key in ("degraded", "models_requested", "models_used", "wp03_run_id")
            if key in payload
        }

    def search(
        self, query_id: str, query_text: str, event_index: int | None = None,
        *, top_k: int | None = None, models: tuple[str, ...] | None = None,
        candidate_frames: list[dict[str, object]] | None = None,
    ) -> list[SearchCandidate]:
        if not self.enabled:
            return []
        result_limit = self.top_k if top_k is None else max(1, int(top_k))
        requested_models = self.models if models is None else tuple(models)
        if self.persistent:
            try:
                payload = self._persistent_request({
                    "operation": "search", "query_id": query_id,
                    "query_text": query_text, "event_index": event_index,
                    "top_k": result_limit,
                    "candidate_k_per_model": self.candidate_k_per_model,
                    "models": ",".join(requested_models),
                    "candidate_frames": candidate_frames or [],
                })
            except TV2VisualClientError as exc:
                self._last_diagnostics = {
                    "degraded": True, "models_requested": [], "models_used": [],
                    "reason": "wp03_persistent_search_failed", "error": str(exc),
                }
                return []
            self._record_diagnostics(payload)
            return [SearchCandidate.from_json(c) for c in payload.get("candidates", [])]
        cmd = [
            self.python_executable, "-m", "wp03", "search",
            "--artifact-root", str(self.artifact_root),
            "--query", query_text,
            "--query-id", query_id,
            "--top-k", str(result_limit),
            "--candidate-k-per-model", str(self.candidate_k_per_model),
        ]
        if event_index is not None:
            cmd += ["--event-index", str(event_index)]
        if self.runtime_root:
            cmd += ["--runtime-root", str(self.runtime_root)]
        if self.runtime_profile:
            cmd += ["--runtime-profile", str(self.runtime_profile)]
        if requested_models:
            cmd += ["--models", ",".join(requested_models)]
        if candidate_frames:
            cmd += ["--candidate-frames-json", json.dumps(candidate_frames, ensure_ascii=False)]
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.wp03_cwd), capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TV2VisualClientError(f"failed to invoke wp03 search: {exc}") from exc
        if proc.returncode != 0:
            # Degrade gracefully: visual branch unavailable (e.g. BEiT-3
            # locked pending validation) should not abort the whole query.
            self._last_diagnostics = {
                "degraded": True,
                "models_requested": [],
                "models_used": [],
                "reason": "wp03_search_failed",
            }
            return []
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise TV2VisualClientError(f"wp03 search returned non-JSON output: {proc.stdout[:300]}") from exc
        self._record_diagnostics(payload)
        candidates = payload.get("candidates", [])
        return [SearchCandidate.from_json(c) for c in candidates]

    def search_batch(
        self, requests: list[tuple[str, str, int | None]], *, top_k: int | None = None,
        models: tuple[str, ...] | None = None,
        candidate_frames: list[dict[str, object]] | None = None,
    ) -> list[list[SearchCandidate]]:
        """Batch visual search: pass list of (query_id, query_text, event_index) and return list of candidate lists."""
        if not self.enabled or not requests:
            return [[] for _ in requests]
        result_limit = self.top_k if top_k is None else max(1, int(top_k))
        requested_models = self.models if models is None else tuple(models)
        queries_payload = [
            {
                "query_id": qid,
                "query_text": qtext,
                "event_index": ev_idx,
                "top_k": result_limit,
            }
            for qid, qtext, ev_idx in requests
        ]
        if self.persistent:
            try:
                payload = self._persistent_request({
                    "operation": "search-batch", "queries": queries_payload,
                    "top_k": result_limit,
                    "candidate_k_per_model": self.candidate_k_per_model,
                    "models": ",".join(requested_models),
                    "candidate_frames": candidate_frames or [],
                })
            except TV2VisualClientError as exc:
                self._last_diagnostics = {
                    "degraded": True, "models_requested": [], "models_used": [],
                    "reason": "wp03_persistent_batch_failed", "error": str(exc),
                }
                return [[] for _ in requests]
            results = payload.get("results", [])
            if isinstance(results, list) and results:
                requested = sorted({model for item in results for model in item.get("models_requested", [])})
                used = sorted({model for item in results for model in item.get("models_used", [])})
                self._last_diagnostics = {
                    "degraded": any(bool(item.get("degraded", False)) for item in results),
                    "models_requested": requested,
                    "models_used": used,
                }
            return [
                [SearchCandidate.from_json(c) for c in item.get("candidates", [])]
                for item in results
            ]
        cmd = [
            self.python_executable, "-m", "wp03", "search-batch",
            "--artifact-root", str(self.artifact_root),
            "--queries-json", json.dumps(queries_payload),
            "--top-k", str(result_limit),
            "--candidate-k-per-model", str(self.candidate_k_per_model),
        ]
        if self.runtime_root:
            cmd += ["--runtime-root", str(self.runtime_root)]
        if self.runtime_profile:
            cmd += ["--runtime-profile", str(self.runtime_profile)]
        if requested_models:
            cmd += ["--models", ",".join(requested_models)]
        if candidate_frames:
            cmd += ["--candidate-frames-json", json.dumps(candidate_frames, ensure_ascii=False)]
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.wp03_cwd), capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TV2VisualClientError(f"failed to invoke wp03 search-batch: {exc}") from exc
        if proc.returncode != 0:
            # Older WP03 deployments expose only ``search``.  TRAKE used to
            # silently discard the complete visual branch in that case, then
            # align unrelated OCR/ASR matches.  Keep the optimisation when
            # available, but preserve correct retrieval by issuing the public
            # single-query command for each item on legacy workers.
            diagnostics = f"{proc.stdout}\n{proc.stderr}".casefold()
            if "search-batch" in diagnostics and "invalid choice" in diagnostics:
                fallback_results: list[list[SearchCandidate]] = []
                for query_id, query_text, event_index in requests:
                    try:
                        candidates = self.search(
                            query_id, query_text, event_index=event_index,
                            top_k=result_limit,
                        )
                    except TypeError:
                        # Preserve compatibility with test doubles and older
                        # in-process clients exposing the original signature.
                        candidates = self.search(
                            query_id, query_text, event_index=event_index,
                        )
                    fallback_results.append(candidates)
                return fallback_results
            self._last_diagnostics = {
                "degraded": True,
                "models_requested": [],
                "models_used": [],
                "reason": "wp03_search_batch_failed",
            }
            return [[] for _ in requests]
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise TV2VisualClientError(f"wp03 search-batch returned non-JSON output: {proc.stdout[:300]}") from exc
        results = payload.get("results", [])
        if results:
            requested = sorted({model for item in results for model in item.get("models_requested", [])})
            used = sorted({model for item in results for model in item.get("models_used", [])})
            self._last_diagnostics = {
                "degraded": any(bool(item.get("degraded", False)) for item in results),
                "models_requested": requested,
                "models_used": used,
            }
        return [
            [SearchCandidate.from_json(c) for c in item.get("candidates", [])]
            for item in results
        ]
