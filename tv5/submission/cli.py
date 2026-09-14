"""CLI fallback tool for validating and packaging AIC contest submissions without a browser."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .validator import validate_csv_file, validate_submission_package
from .packager import package_submission_zip
from .from_predictions_final import PredictionsFinalError, build_submission_zip_from_predictions_final


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tv5.submission",
        description="WP13 Contest Submission Validation & Packaging CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-csv
    p_val_csv = subparsers.add_parser("validate-csv", help="Validate a per-query CSV file")
    p_val_csv.add_argument("path", type=Path, help="Path to CSV file")
    p_val_csv.add_argument("--task-type", choices=["KIS", "VQA", "TRAKE"], default=None)

    # validate-package
    p_val_pkg = subparsers.add_parser("validate-package", help="Validate a submission ZIP package")
    p_val_pkg.add_argument("path", type=Path, help="Path to ZIP package")

    # package
    p_pkg = subparsers.add_parser("package", help="Package a directory of CSVs into submission ZIP")
    p_pkg.add_argument("input_dir", type=Path, help="Directory containing query CSV files")
    p_pkg.add_argument("output_zip", type=Path, help="Output ZIP file path")

    # from-predictions-final
    p_from_pf = subparsers.add_parser(
        "from-predictions-final",
        help="Build a submission ZIP directly from the Retrieval Owner's predictions_final.json",
    )
    p_from_pf.add_argument("predictions_final", type=Path, help="Path to predictions_final.json")
    p_from_pf.add_argument("output_zip", type=Path, help="Output ZIP file path")

    # check-api
    p_api = subparsers.add_parser("check-api", help="Check status of automated competition upload API")

    parsed = parser.parse_args(args)

    if parsed.command == "validate-csv":
        report = validate_csv_file(parsed.path, task_type=parsed.task_type)
        print(f"Validation: {'PASS' if report.is_valid else 'FAIL'}")
        print(f"Task Type: {report.task_type}")
        print(f"Record Count: {report.record_count}")
        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  - {e}")
        return 0 if report.is_valid else 1

    elif parsed.command == "validate-package":
        report = validate_submission_package(parsed.path)
        print(f"Package Validation: {'PASS' if report.is_valid else 'FAIL'}")
        print(f"SHA-256 Digest: {report.package_digest_sha256}")
        print(f"CSV Files Count: {report.record_count}")
        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  - {e}")
        return 0 if report.is_valid else 1

    elif parsed.command == "package":
        if not parsed.input_dir.is_dir():
            print(f"Error: {parsed.input_dir} is not a directory", file=sys.stderr)
            return 1
        csv_files = {p.stem: p for p in parsed.input_dir.glob("*.csv")}
        if not csv_files:
            print(f"Error: no .csv files found in {parsed.input_dir}", file=sys.stderr)
            return 1
        report = package_submission_zip(csv_files, parsed.output_zip)
        print(f"Package Created: {parsed.output_zip}")
        print(f"Validation: {'PASS' if report.is_valid else 'FAIL'}")
        print(f"SHA-256 Digest: {report.package_digest_sha256}")
        print(f"Contained CSVs: {report.record_count}")
        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  - {e}")
        return 0 if report.is_valid else 1

    elif parsed.command == "from-predictions-final":
        try:
            report = build_submission_zip_from_predictions_final(parsed.predictions_final, parsed.output_zip)
        except PredictionsFinalError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"Package Created: {parsed.output_zip}")
        print(f"Validation: {'PASS' if report.is_valid else 'FAIL'}")
        print(f"SHA-256 Digest: {report.package_digest_sha256}")
        print(f"Contained CSVs: {report.record_count}")
        if report.errors:
            print("Errors:")
            for e in report.errors:
                print(f"  - {e}")
        return 0 if report.is_valid else 1

    elif parsed.command == "check-api":
        print("STATUS: UNAVAILABLE")
        print("INFO: No official competition direct-upload API contract exists.")
        print("POLICY: Competition submission ZIP upload remains strictly human-controlled.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
