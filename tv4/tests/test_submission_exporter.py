import json

import pytest

from tv4.contracts import EvidencePack, SearchCandidate, TrakeHypothesis
from tv4.submission_exporter import (
    AuthoritativeFrameResolver,
    PredictionValidationError,
    QuerySpec,
    build_predictions_final,
    validate_predictions,
    validate_resolved_predictions,
    write_predictions_final,
)
from tv4.wp11_vqa import VqaResult


class _TV1:
    def frames(self, video_id):
        return [
            {"frame_id": 10, "timestamp_ms": 1000},
            {"frame_id": 20, "timestamp_ms": 2000},
            {"frame_id": 30, "timestamp_ms": 3000},
        ]


def _candidate(frame_id=10):
    return SearchCandidate(
        query_id="q", video_id="L21_V005", frame_id=frame_id, timestamp_ms=frame_id * 100,
        source="fusion", rank=1,
    )


def _vqa_result(frame_id=20, answer="5"):
    candidate = _candidate(frame_id)
    evidence = EvidencePack("q", candidate.video_id, frame_id, candidate.timestamp_ms, None)
    return VqaResult(
        candidate, evidence, answer, verified=True, manual_fallback=False,
        proposal=answer, approved=True, verifier_status="verified",
        retry_count=0, manual_required=False, status="proposal",
    )


def test_exporter_writes_the_minimal_final_contract(tmp_path):
    specs = [
        QuerySpec("kis-1", "KIS"),
        QuerySpec("vqa-1", "VQA"),
        QuerySpec("trake-1", "TRAKE", event_count=2),
    ]
    trake = TrakeHypothesis("trake-1", "L21_V005", (10, 30), (0.9, 0.8), 1.7, "run")
    output = tmp_path / "predictions_final.json"
    summary = write_predictions_final(
        output,
        specs,
        {"kis-1": [_candidate(10)], "vqa-1": [_vqa_result(20)], "trake-1": trake},
        AuthoritativeFrameResolver(_TV1()),
    )

    assert summary.query_count == 3
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "kis-1": [{"video": "L21_V005", "frame": 10}],
        "vqa-1": [{"video": "L21_V005", "frame": 20, "answer": "5"}],
        "trake-1": [{"video": "L21_V005", "frames": [10, 30]}],
    }


def test_exporter_rejects_a_frame_not_known_by_tv1():
    with pytest.raises(PredictionValidationError, match="authoritative TV1 mapping"):
        build_predictions_final(
            [QuerySpec("kis-1", "KIS")],
            {"kis-1": [_candidate(999)]},
            AuthoritativeFrameResolver(_TV1()),
        )


def test_independent_validator_rechecks_final_frame_membership():
    predictions = {"kis-1": [{"video": "L21_V005", "frame": 999}]}
    with pytest.raises(PredictionValidationError, match="authoritative TV1 mapping"):
        validate_resolved_predictions(
            predictions, [QuerySpec("kis-1", "KIS")], AuthoritativeFrameResolver(_TV1())
        )


@pytest.mark.parametrize(
    "prediction",
    [
        {"q": [{"video": "L21_V005.mp4", "frame": 10}]},
        {"q": [{"video": "L21_V005", "frame": "10"}]},
        {"q": [{"video": "L21_V005", "frames": [10, 10]}]},
    ],
)
def test_validator_rejects_unsafe_submission_values(prediction):
    task = "TRAKE" if "frames" in prediction["q"][0] else "KIS"
    spec = QuerySpec("q", task, event_count=2 if task == "TRAKE" else None)
    with pytest.raises(PredictionValidationError):
        validate_predictions(prediction, [spec])


def test_exporter_rejects_manual_review_vqa():
    unsafe = _vqa_result()
    unsafe = VqaResult(
        unsafe.candidate, unsafe.evidence, unsafe.answer, verified=False, manual_fallback=True,
        proposal=unsafe.answer, approved=False, verifier_status="rejected",
        retry_count=0, manual_required=True, status="manual_required",
    )
    with pytest.raises(PredictionValidationError, match="manual review"):
        build_predictions_final(
            [QuerySpec("vqa-1", "VQA")],
            {"vqa-1": [unsafe]},
            AuthoritativeFrameResolver(_TV1()),
        )
