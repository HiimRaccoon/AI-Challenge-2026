"""Subprocess client for TV2's WP09 exact-frame refinement JSON CLI.

TV4 provides the request payload (coarse candidate + task + refinement text
+ policy) via a temp JSON file and reads back one `RefineResult` JSON object
from stdout, per WP09's documented CLI contract:

    python -m wp09 refine --request request.json --config configs/default.yaml \
        --decoder-factory <module:callable> --scorer-factory <module:callable>

`--decoder-factory` / `--scorer-factory` point at TV4's own adapter
(`tv4.adapters.wp09_adapter`) because no other team member has published one
yet; see that module for the implementation.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


class TV2RefineClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExactProofAuthority:
    certification_id: str
    certification_report_sha256: str
    preprocess_run_id: str

    @classmethod
    def load(cls, path: Path) -> "ExactProofAuthority":
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = (raw.get("certification_id"), raw.get("certification_report_sha256"), raw.get("preprocess_run_id")) if isinstance(raw, dict) else ()
        if len(values) != 3 or not all(isinstance(value, str) and value for value in values):
            raise TV2RefineClientError("invalid exact certification authority")
        return cls(*values)


@dataclass
class TV2RefineClient:
    python_executable: str
    wp09_cwd: Path
    config_path: Path
    decoder_factory: str = "tv4.adapters.wp09_adapter:decoder_for_request"
    scorer_factory: str | None = "tv4.adapters.wp09_adapter:scorer_for_request"
    resolver_factory: str = "tv4.adapters.wp09_adapter:resolver_for_request"
    tv1_base_url: str | None = None
    exact_certification_path: Path | None = None
    enabled: bool = True
    last_error: str | None = field(default=None, init=False, repr=False)

    def health(self) -> dict[str, object]:
        checks = {
            "python_executable": Path(self.python_executable).is_file(),
            "wp09_cwd": self.wp09_cwd.is_dir(),
            "config_path": self.config_path.is_file(),
            "exact_certification": bool(
                self.exact_certification_path and self.exact_certification_path.is_file()
            ),
        }
        return {
            "available": bool(self.enabled and all(checks.values())),
            "enabled": self.enabled,
            "checks": checks,
            "last_error": self.last_error,
        }

    @property
    def exact_proof_authority(self) -> ExactProofAuthority | None:
        return ExactProofAuthority.load(self.exact_certification_path) if self.exact_certification_path else None

    def refine(self, request_payload: dict) -> dict | None:
        if not self.enabled:
            return None
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(request_payload, fh, ensure_ascii=False)
            request_path = Path(fh.name)
        try:
            cmd = [
                self.python_executable, "-m", "wp09", "refine",
                "--request", str(request_path),
                "--config", str(self.config_path),
                "--decoder-factory", self.decoder_factory,
            ]
            if self.scorer_factory:
                cmd += ["--scorer-factory", self.scorer_factory]
            try:
                proc = subprocess.run(
                    cmd, cwd=str(self.wp09_cwd), capture_output=True, text=True, timeout=60, env=self._subprocess_env(),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise TV2RefineClientError(f"failed to invoke wp09 refine: {exc}") from exc
            if proc.returncode != 0:
                self.last_error = (proc.stderr or proc.stdout or "wp09 refine failed")[:1000]
                return None
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return None
        finally:
            request_path.unlink(missing_ok=True)

    def neighbors(self, request_payload: dict) -> dict | None:
        """Run WP09's narrow neighbor contract without a raw-ID fallback."""
        if not self.enabled:
            return None
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(request_payload, fh, ensure_ascii=False)
            request_path = Path(fh.name)
        try:
            try:
                proc = subprocess.run(
                    [self.python_executable, "-m", "wp09", "neighbors", "--request", str(request_path), "--resolver-factory", self.resolver_factory],
                    cwd=str(self.wp09_cwd), capture_output=True, text=True, timeout=60, env=self._subprocess_env(),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"[:1000]
                return None
            if proc.returncode != 0:
                self.last_error = (proc.stderr or proc.stdout or "wp09 neighbors failed")[:1000]
                return None
            try:
                result = json.loads(proc.stdout)
                self.last_error = None
                return result
            except json.JSONDecodeError:
                self.last_error = "wp09 neighbors returned invalid JSON"
                return None
        finally:
            request_path.unlink(missing_ok=True)

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # The subprocess runs from the WP09 project root, while both projects
        # use a src/ layout.  Make their installed-source roots explicit so a
        # clean production venv does not accidentally depend on a developer's
        # ambient PYTHONPATH.
        roots = [str(self.wp09_cwd / "src"), str(Path(__file__).resolve().parents[2])]
        inherited = env.get("PYTHONPATH")
        if inherited:
            roots.append(inherited)
        env["PYTHONPATH"] = os.pathsep.join(roots)
        if self.tv1_base_url:
            env["TV4_TV1_BASE_URL"] = self.tv1_base_url
        if self.exact_certification_path:
            env["TV4_EXACT_CERTIFICATION_PATH"] = str(self.exact_certification_path)
        return env
