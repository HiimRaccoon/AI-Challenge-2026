"""RFC 4180 compliant, headerless UTF-8 CSV exporter and parser for KIS, VQA, and TRAKE."""
from __future__ import annotations

import csv
import io
from typing import Sequence
from .contracts import KisPrediction, VqaPrediction, TrakePrediction


def export_kis_csv(predictions: Sequence[KisPrediction]) -> str:
    """Export up to 100 KIS predictions as headerless CSV: <video_id>,<frame_id>."""
    if len(predictions) > 100:
        raise ValueError(f"KIS export exceeds limit of 100 predictions (got {len(predictions)})")

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for p in predictions:
        errs = p.validate()
        if errs:
            raise ValueError(f"Invalid KIS prediction: {', '.join(errs)}")
        writer.writerow([p.video_id, str(p.frame_id)])
    return out.getvalue()


def export_vqa_csv(predictions: Sequence[VqaPrediction]) -> str:
    """Export up to 100 VQA predictions as headerless CSV: <video_id>,<frame_id>,<approved_answer>."""
    if len(predictions) > 100:
        raise ValueError(f"VQA export exceeds limit of 100 predictions (got {len(predictions)})")

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for p in predictions:
        errs = p.validate()
        if errs:
            raise ValueError(f"Invalid VQA prediction: {', '.join(errs)}")
        writer.writerow([p.video_id, str(p.frame_id), p.approved_answer])
    return out.getvalue()


def export_trake_csv(predictions: Sequence[TrakePrediction]) -> str:
    """Export up to 100 TRAKE predictions as headerless CSV: <video_id>,<frame_id_1>,...,<frame_id_N>."""
    if len(predictions) > 100:
        raise ValueError(f"TRAKE export exceeds limit of 100 predictions (got {len(predictions)})")

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for p in predictions:
        errs = p.validate()
        if errs:
            raise ValueError(f"Invalid TRAKE prediction: {', '.join(errs)}")
        row = [p.video_id] + [str(fid) for fid in p.event_frame_ids]
        writer.writerow(row)
    return out.getvalue()


def parse_submission_csv(csv_text: str) -> list[list[str]]:
    """Parse submission CSV text into rows, adhering to RFC 4180 rules."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        if row:  # skip empty lines if any
            rows.append(row)
    return rows
