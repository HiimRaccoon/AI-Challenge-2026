"""Submission package builder with automatic reopen-validation and digest calculation."""
from __future__ import annotations

from pathlib import Path
import zipfile
from typing import Mapping

from .validator import validate_submission_package, ValidationReport


def package_submission_zip(
    query_csv_files: Mapping[str, Path | str],
    output_zip_path: Path,
) -> ValidationReport:
    """Package query CSV files into top-level submission/ ZIP, then immediately reopen and validate."""
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for query_name, file_or_content in query_csv_files.items():
            filename = query_name if query_name.endswith(".csv") else f"{query_name}.csv"
            arcname = f"submission/{filename}"
            if isinstance(file_or_content, Path):
                zf.write(file_or_content, arcname=arcname)
            else:
                zf.writestr(arcname, file_or_content.encode("utf-8"))

    # Fail-closed: reopen archive and validate
    report = validate_submission_package(output_zip_path)
    return report
