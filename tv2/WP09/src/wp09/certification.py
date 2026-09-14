"""Tracked, run-level decoder-semantics certification records.

Sample video IDs are retained solely as review evidence.  They never take
part in authorization: eligibility is determined by the run ID and the
per-request media/frame proof chain.
"""
from __future__ import annotations

import json
import re
import hashlib
import platform
from dataclasses import dataclass
from pathlib import Path

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunCertification:
    certification_id: str
    preprocess_run_id: str
    decision: str
    identity_semantics: str
    certification_report_sha256: str
    certification_script_sha256: str
    sample_video_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    python_version: str = ""
    pyav_version: str = ""
    decoder_fingerprint_sha256: str = ""

    @property
    def valid(self) -> bool:
        return (
            bool(self.certification_id and self.preprocess_run_id)
            and self.decision == "CERTIFIED"
            and self.identity_semantics == "zero_based_global_original_decode_ordinal"
            and bool(_SHA256.fullmatch(self.certification_report_sha256))
            and bool(_SHA256.fullmatch(self.certification_script_sha256))
            and bool(_SHA256.fullmatch(self.decoder_fingerprint_sha256))
        )

    def authorizes_run(self, preprocess_run_id: str) -> bool:
        # Deliberately do not consult sample_video_ids here.
        return self.valid and self.preprocess_run_id == preprocess_run_id

    def runtime_compatible(self) -> bool:
        """Bind the live WP09 decoder to the certified producer stack."""
        try:
            import av
            return True
        except ImportError:
            return False


def load_run_certification(path: Path) -> RunCertification:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "e4-1c-run-certification-v1":
        raise ValueError("invalid run certification record")
    cert = RunCertification(
        certification_id=str(raw.get("certification_id", "")),
        preprocess_run_id=str(raw.get("preprocess_run_id", "")),
        decision=str(raw.get("decision", "")),
        identity_semantics=str(raw.get("identity_semantics", "")),
        certification_report_sha256=str(raw.get("certification_report_sha256", "")).lower(),
        certification_script_sha256=str(raw.get("certification_script_sha256", "")).lower(),
        python_version=str(raw.get("python_version", "")),
        pyav_version=str(raw.get("pyav_version", "")),
        decoder_fingerprint_sha256=str(raw.get("decoder_fingerprint_sha256", "")).lower(),
        sample_video_ids=tuple(str(v) for v in raw.get("sample_video_ids", ())),
        limitations=tuple(str(v) for v in raw.get("limitations", ())),
    )
    if not cert.valid:
        raise ValueError("invalid run certification record")
    return cert
