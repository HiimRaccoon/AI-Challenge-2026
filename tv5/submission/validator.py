"""Fail-closed validator for predictions, CSV submission files, and submission ZIP packages."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import re
import zipfile
from typing import Literal

from .contracts import KisPrediction, VqaPrediction, TrakePrediction, ValidationReport, TaskType

HEADER_KEYWORDS = {"video_id", "frame_id", "answer", "video", "frame", "rank", "score", "timestamp"}


def validate_prediction(prediction: KisPrediction | VqaPrediction | TrakePrediction) -> ValidationReport:
    errs = prediction.validate()
    task: TaskType = "KIS"
    if isinstance(prediction, VqaPrediction):
        task = "VQA"
    elif isinstance(prediction, TrakePrediction):
        task = "TRAKE"
    return ValidationReport(is_valid=not errs, errors=tuple(errs), task_type=task, record_count=1)


def validate_csv_file(path: Path, task_type: TaskType | None = None) -> ValidationReport:
    if not path.exists():
        return ValidationReport(is_valid=False, errors=(f"File does not exist: {path}",))

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return ValidationReport(is_valid=False, errors=(f"Failed to read file: {exc}",))

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return ValidationReport(is_valid=False, errors=(f"File is not valid UTF-8: {exc}",))

    reader = csv.reader(io.StringIO(text))
    rows: list[list[str]] = []
    for r in reader:
        if r:
            rows.append(r)

    if not rows:
        return ValidationReport(is_valid=False, errors=("CSV file is empty",), record_count=0)

    if len(rows) > 100:
        return ValidationReport(
            is_valid=False,
            errors=(f"CSV file contains {len(rows)} rows, exceeding maximum limit of 100",),
            record_count=len(rows),
        )

    # Detect header row
    first_row = [c.strip().lower() for c in rows[0]]
    if any(c in HEADER_KEYWORDS for c in first_row):
        return ValidationReport(
            is_valid=False,
            errors=("CSV file contains forbidden header row (submissions must be headerless)",),
            record_count=len(rows),
        )

    # Inferred or explicit task
    inferred_task = task_type
    if inferred_task is None:
        first_len = len(rows[0])
        if first_len == 2:
            inferred_task = "KIS"
        elif first_len == 3:
            # could be VQA or 2-event TRAKE; check if 3rd col is integer
            try:
                int(rows[0][2].strip())
                inferred_task = "TRAKE"
            except ValueError:
                inferred_task = "VQA"
        elif first_len > 3:
            inferred_task = "TRAKE"
        else:
            return ValidationReport(
                is_valid=False,
                errors=(f"Row 1 has invalid column count: {first_len}",),
                record_count=len(rows),
            )

    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        if not row:
            continue
        vid = row[0].strip()
        if not vid:
            errors.append(f"Row {idx}: missing video_id")
        elif vid.endswith(".mp4"):
            errors.append(f"Row {idx}: video_id '{vid}' contains forbidden .mp4 extension")

        if inferred_task == "KIS":
            if len(row) != 2:
                errors.append(f"Row {idx}: KIS requires exactly 2 columns (<video_id>,<frame_id>), got {len(row)}")
            else:
                try:
                    fid = int(row[1].strip())
                    if fid < 0:
                        errors.append(f"Row {idx}: frame_id must be non-negative integer, got {fid}")
                except ValueError:
                    errors.append(f"Row {idx}: frame_id must be integer, got '{row[1]}'")

        elif inferred_task == "VQA":
            if len(row) != 3:
                errors.append(f"Row {idx}: VQA requires exactly 3 columns (<video_id>,<frame_id>,<answer>), got {len(row)}")
            else:
                try:
                    fid = int(row[1].strip())
                    if fid < 0:
                        errors.append(f"Row {idx}: frame_id must be non-negative integer, got {fid}")
                except ValueError:
                    errors.append(f"Row {idx}: frame_id must be integer, got '{row[1]}'")
                answer = row[2]
                if not answer.strip():
                    errors.append(f"Row {idx}: VQA approved_answer must not be blank")
                elif len(answer) > 100:
                    errors.append(f"Row {idx}: VQA approved_answer exceeds 100 characters ({len(answer)})")

        elif inferred_task == "TRAKE":
            if len(row) < 3:
                errors.append(f"Row {idx}: TRAKE requires at least 3 columns (<video_id>,<f1>,<f2>...), got {len(row)}")
            else:
                for f_idx, val in enumerate(row[1:], start=1):
                    try:
                        fid = int(val.strip())
                        if fid < 0:
                            errors.append(f"Row {idx}, Event {f_idx}: frame_id must be non-negative integer, got {fid}")
                    except ValueError:
                        errors.append(f"Row {idx}, Event {f_idx}: frame_id must be integer, got '{val}'")

    return ValidationReport(
        is_valid=not errors,
        errors=tuple(errors),
        task_type=inferred_task,
        record_count=len(rows),
    )


def validate_submission_package(zip_path: Path) -> ValidationReport:
    """Validate submission ZIP archive structure and every contained query CSV."""
    if not zip_path.exists():
        return ValidationReport(is_valid=False, errors=(f"Package does not exist: {zip_path}",))

    try:
        raw_bytes = zip_path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
    except OSError as exc:
        return ValidationReport(is_valid=False, errors=(f"Cannot read package: {exc}",))

    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
            namelist = zf.namelist()
            if not namelist:
                return ValidationReport(is_valid=False, errors=("Package archive is empty",), package_digest_sha256=digest)

            errors: list[str] = []
            csv_count = 0

            for name in namelist:
                # Check traversal
                if name.startswith("/") or name.startswith("\\") or ".." in name:
                    errors.append(f"Forbidden path traversal in archive member: {name}")
                    continue

                if name.endswith("/"):
                    # Directory entry
                    if not name.startswith("submission/"):
                        errors.append(f"Forbidden directory outside submission/: {name}")
                    continue

                # File entry
                if not name.startswith("submission/"):
                    errors.append(f"File outside required top-level 'submission/' directory: {name}")
                    continue

                rel_name = name[len("submission/"):]
                if "/" in rel_name or "\\" in rel_name:
                    errors.append(f"Nested subdirectory not allowed inside submission/: {name}")
                    continue

                if not rel_name.endswith(".csv"):
                    errors.append(f"Unexpected non-CSV file inside submission/: {name}")
                    continue

                csv_count += 1
                try:
                    content_bytes = zf.read(name)
                    text = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    errors.append(f"File {name} is not valid UTF-8")
                    continue

                # In-memory CSV validation
                rdr = csv.reader(io.StringIO(text))
                rows = [r for r in rdr if r]
                if not rows:
                    errors.append(f"File {name} is empty")
                elif len(rows) > 100:
                    errors.append(f"File {name} contains {len(rows)} rows (>100 limit)")
                elif any(c.strip().lower() in HEADER_KEYWORDS for c in rows[0]):
                    errors.append(f"File {name} contains forbidden header row")

            if csv_count == 0 and not errors:
                errors.append("Package contains no query CSV files under submission/")

            return ValidationReport(
                is_valid=not errors,
                errors=tuple(errors),
                record_count=csv_count,
                package_digest_sha256=digest,
            )

    except zipfile.BadZipFile as exc:
        return ValidationReport(is_valid=False, errors=(f"Malformed ZIP file: {exc}",), package_digest_sha256=digest)
