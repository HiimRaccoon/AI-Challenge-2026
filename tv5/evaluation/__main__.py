"""CLI runner for evaluation metrics and report generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import calculate_final_score, calculate_r_at_k, evaluate_kis_prediction, GroundTruthInterval
from .preprocessing_reports import ingest_preprocessing_run_report
from .reports import BenchmarkReport


def main() -> int:
    parser = argparse.ArgumentParser(description="WP13 Evaluation & Metric CLI Tool")
    parser.add_argument("--benchmark", action="store_true", help="Print benchmark metric formulas and golden report")
    parser.add_argument("--run-dir", type=str, help="Ingest and print summary of a preprocessing run directory")
    args = parser.parse_args()

    if args.run_dir:
        path = Path(args.run_dir)
        report = ingest_preprocessing_run_report(path)
        print(f"Preprocessing Report ({report.run_id}): status={report.status}, videos={report.video_count}, valid={report.is_valid}")
        return 0

    # Default: print benchmark golden simulation
    golden_r = {1: 0.45, 5: 0.68, 20: 0.82, 50: 0.89, 100: 0.94}
    final_score = calculate_final_score(golden_r)
    print(f"=== AIC 2026 Evaluation Engine (T046) ===")
    print(f"Final Score (mean of R@k): {final_score:.4f} (Golden = 0.7400)")
    print(f"R@k Distribution: {golden_r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
