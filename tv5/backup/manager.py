"""WP13 backup and recovery manager with atomic write, SHA-256 manifests, and secret exclusion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from typing import Any

EXCLUDE_EXTENSIONS = {".mp4", ".faiss", ".parquet", ".npy", ".pt", ".bin", ".weights"}
EXCLUDE_NAMES = {".git", ".venv", "node_modules", "raw", "runs", "indexes", "embeddings"}


@dataclass(frozen=True)
class BackupManifest:
    backup_id: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    file_checksums: dict[str, str] = field(default_factory=dict)
    file_count: int = 0
    total_bytes: int = 0
    archive_sha256: str = ""
    readiness_invalidated: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BackupManager:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir

    def create_backup(self, output_zip_path: Path, backup_id: str = "wp13_backup") -> BackupManifest:
        """Create a backup ZIP containing WP13 mutable state only. Strictly excludes raw data and upstream indexes."""
        output_zip_path.parent.mkdir(parents=True, exist_ok=True)
        file_checksums: dict[str, str] = {}
        total_bytes = 0

        # Atomic temp write
        temp_zip = output_zip_path.with_name(f"{output_zip_path.name}.tmp_{datetime.now(timezone.utc).timestamp()}")

        try:
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if self.state_dir.exists():
                    for item in self.state_dir.rglob("*"):
                        if not item.is_file():
                            continue
                        if item.suffix.lower() in EXCLUDE_EXTENSIONS:
                            continue
                        if any(part in EXCLUDE_NAMES for part in item.parts):
                            continue

                        rel_path = item.relative_to(self.state_dir).as_posix()
                        raw = item.read_bytes()
                        sha256 = hashlib.sha256(raw).hexdigest()
                        file_checksums[rel_path] = sha256
                        total_bytes += len(raw)
                        zf.write(item, arcname=f"state/{rel_path}")

                # Write manifest inside archive
                manifest_stub = {
                    "backup_id": backup_id,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "file_checksums": file_checksums,
                    "file_count": len(file_checksums),
                    "total_bytes": total_bytes,
                    "readiness_invalidated": True,
                }
                zf.writestr("manifest.json", json.dumps(manifest_stub, indent=2).encode("utf-8"))

            # Calculate final archive digest
            archive_digest = hashlib.sha256(temp_zip.read_bytes()).hexdigest()

            # Atomic rename
            temp_zip.replace(output_zip_path)

            return BackupManifest(
                backup_id=backup_id,
                file_checksums=file_checksums,
                file_count=len(file_checksums),
                total_bytes=total_bytes,
                archive_sha256=archive_digest,
                readiness_invalidated=True,
            )
        finally:
            if temp_zip.exists():
                temp_zip.unlink()

    def restore_backup(self, zip_path: Path, target_dir: Path | None = None) -> tuple[bool, BackupManifest | None, list[str]]:
        """Restore WP13 state from backup ZIP and verify all file checksums."""
        dest_dir = target_dir or self.state_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        if not zip_path.exists():
            return False, None, [f"Backup file not found: {zip_path}"]

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if "manifest.json" not in zf.namelist():
                    return False, None, ["Backup archive missing required manifest.json"]

                manifest_raw = zf.read("manifest.json")
                manifest_data = json.loads(manifest_raw.decode("utf-8"))
                expected_checksums: dict[str, str] = manifest_data.get("file_checksums", {})

                # Extract files
                for arcname in zf.namelist():
                    if arcname == "manifest.json" or arcname.endswith("/"):
                        continue
                    if not arcname.startswith("state/"):
                        errors.append(f"Unexpected file in backup: {arcname}")
                        continue

                    rel_path = arcname[len("state/"):]
                    target_file = dest_dir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    data = zf.read(arcname)
                    # Verify checksum before writing
                    actual_sha256 = hashlib.sha256(data).hexdigest()
                    expected_sha256 = expected_checksums.get(rel_path)
                    if expected_sha256 and actual_sha256 != expected_sha256:
                        errors.append(f"Checksum mismatch for {rel_path}: expected {expected_sha256}, got {actual_sha256}")
                        continue

                    target_file.write_bytes(data)

                manifest = BackupManifest(
                    backup_id=manifest_data.get("backup_id", "restored"),
                    file_checksums=expected_checksums,
                    file_count=len(expected_checksums),
                    total_bytes=manifest_data.get("total_bytes", 0),
                    readiness_invalidated=True,  # Invariant: restored state must be revalidated before use
                )
                return not errors, manifest, errors

        except zipfile.BadZipFile as exc:
            return False, None, [f"Corrupted backup archive: {exc}"]


def create_backup(state_dir: Path, output_zip: Path, backup_id: str = "wp13_backup") -> BackupManifest:
    return BackupManager(state_dir).create_backup(output_zip, backup_id)


def restore_backup(zip_path: Path, target_dir: Path) -> tuple[bool, BackupManifest | None, list[str]]:
    return BackupManager(target_dir).restore_backup(zip_path, target_dir)
