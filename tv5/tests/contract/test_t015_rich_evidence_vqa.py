"""T015 rich-evidence / VQA boundary tests, repaired by T016."""

from __future__ import annotations

from dataclasses import fields
import inspect
import json
from pathlib import Path
import sys

import pytest


TV5_ROOT = Path(__file__).resolve().parents[3]
TV4_SRC = TV5_ROOT / "tv4" / "src"
WP04_CONTRACTS = TV5_ROOT.parent / "tv1tv3" / "TV1_TV3_WP04" / "src" / "aic2026" / "contracts.py"
WP04_API = TV5_ROOT.parent / "tv1tv3" / "TV1_TV3_WP04" / "src" / "aic2026" / "api.py"
GOLDEN = Path(__file__).with_name("t015_vqa_evidence_boundary_golden.json")

if str(TV4_SRC) not in sys.path:
    sys.path.insert(0, str(TV4_SRC))

from tv4.contracts import EvidencePack, SearchCandidate, SearchRequest
from tv4.wp11_vqa import RuleBasedFallbackEngine, VqaResult, answer_query, build_evidence_pack


class _FramesOnlyTV1:
    def frames(self, video_id: str) -> list[dict[str, object]]:
        assert video_id == "V001"
        return []


class _CandidateFrameTV1:
    def frames(self, video_id: str) -> list[dict[str, object]]:
        assert video_id == "V001"
        return [{"video_id": "V001", "frame_id": 120, "timestamp_ms": 4_800, "keyframe_path": "fixture/V001/120.jpg"}]


class _RecordingTV3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SearchRequest]] = []

    def search(self, route: str, request: SearchRequest) -> list[SearchCandidate]:
        self.calls.append((route, request))
        return []


class _AlwaysAnswersEngine:
    def answer(self, question: str, evidence: EvidencePack) -> str:
        return "fabricated answer"

    def verify(self, question: str, answer: str, evidence: EvidencePack) -> bool:
        return True


class _UnavailableTV3:
    def search(self, route: str, request: SearchRequest) -> list[SearchCandidate]:
        raise RuntimeError("WP04 unavailable")


class _RejectingEngine:
    def __init__(self) -> None:
        self.answer_calls = 0

    def answer(self, question: str, evidence: EvidencePack) -> str:
        self.answer_calls += 1
        return "uncertain"

    def verify(self, question: str, answer: str, evidence: EvidencePack) -> bool:
        return False


def _candidate() -> SearchCandidate:
    return SearchCandidate(
        query_id="t015-vqa-001",
        video_id="V001",
        frame_id=120,
        timestamp_ms=4_800,
        source="fusion",
        rank=1,
        preprocess_run_id="run-t015",
        provenance_sources=("visual", "ocr", "asr"),
    )


def test_t015_golden_is_machine_readable_and_uses_allowed_classifications() -> None:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["task"] == "T015"
    assert data["current_tv4_contract"]["request"]["path"] == "POST /vqa/answer"
    assert data["classification_at_t015"]["wp04_producer_source_and_contracts"] == "SUPPORTED"
    assert data["classification_at_t015"]["tv4_rich_evidence_mapping"] == "CODE GAP"
    assert data["classification_at_t015"]["wp04_corrected_modality_artifacts"] == "HANDOVER PENDING"
    assert data["t016_repair_status"]["tv4_rich_evidence_mapping"] == "SUPPORTED"
    assert data["t016_repair_status"]["vlm_execution"] == "SCOPED DEGRADATION"
    assert set(data["error_degraded_matrix"]) == {
        "malformed_request", "unknown_canonical_candidate", "empty_evidence", "partial_evidence",
        "wp04_unavailable", "vlm_or_verifier_unavailable", "reasoning_failure",
        "invalid_evidence_schema", "retry_exhausted",
    }
    assert set(data["classification_at_t015"].values()) <= {
        "SUPPORTED", "CODE GAP", "HANDOVER PENDING", "SCOPED DEGRADATION",
        "INCOMPATIBLE", "ACTUALLY MISSING",
    }


def test_wp04_source_contract_has_rich_records_and_catalogue_detail_routes() -> None:
    contracts = WP04_CONTRACTS.read_text(encoding="utf-8")
    api = WP04_API.read_text(encoding="utf-8")
    for field_name in (
        "class OCRDetection", "raw_text", "bbox_xyxy_norm", "polygon_norm",
        "class ASRSegment", "start_ms", "end_ms", "words",
        "class ObjectDetection", "source_keyframe_path", "class MetadataRecord",
        "source_record_sha256",
    ):
        assert field_name in contracts
    for route in (
        '"/ocr/detections"', '"/asr/segments"', '"/asr/{segment_id}/context"',
        '"/objects/detections"', '"/metadata/records"',
    ):
        assert route in api


def test_tv4_probe_preserves_query_and_question__contract() -> None:
    tv3 = _RecordingTV3()
    request = SearchRequest(query_id="t015-vqa-001", task="VQA", query_text="award ceremony", question="What color is the cup?", limit=20)
    build_evidence_pack(_candidate(), _FramesOnlyTV1(), tv3, request=request)
    assert [route for route, _ in tv3.calls] == ["ocr", "asr", "object", "metadata"]
    assert all(call.query_text == request.query_text and call.question == request.question for _, call in tv3.calls)


def test_current_multiframe_ids_are_taken_from_upstream_records_not_local_arithmetic() -> None:
    source = inspect.getsource(build_evidence_pack)
    assert "tv1.frames(candidate.video_id)" in source
    assert "candidate.frame_id +" not in source
    assert "candidate.frame_id -" not in source
    assert "fps" not in source.casefold()


def test_target_propagates_distinct_query_and_question_to_each_wp04_branch() -> None:
    tv3 = _RecordingTV3()
    request = SearchRequest(
        query_id="t015-vqa-001",
        task="VQA",
        query_text="award ceremony with a blue cup",
        question="What color is the cup?",
        limit=20,
    )
    build_evidence_pack(_candidate(), _FramesOnlyTV1(), tv3, request=request)
    assert [route for route, _ in tv3.calls] == ["ocr", "asr", "object", "metadata"]
    assert all(call.query_text == request.query_text for _, call in tv3.calls)
    assert all(call.question == request.question for _, call in tv3.calls)


def test_target_evidence_pack_retains_rich_branch_records_and_provenance() -> None:
    names = {field.name for field in fields(EvidencePack)}
    assert {
        "query_text", "question", "selected_frames", "ocr_evidence", "asr_evidence",
        "object_evidence", "metadata_evidence", "provenance",
    } <= names


def test_target_rule_verifier_rejects_arbitrary_nonempty_answer() -> None:
    evidence = EvidencePack(
        query_id="t015-vqa-001", video_id="V001", frame_id=120, timestamp_ms=4_800,
        keyframe_path=None, ocr_texts=("unrelated sign",),
    )
    assert RuleBasedFallbackEngine().verify("What color is the cup?", "unrelated sign", evidence) is False


def test_target_empty_evidence_never_becomes_verified_or_nonmanual() -> None:
    result = answer_query(
        _candidate(), "What color is the cup?", _FramesOnlyTV1(), _RecordingTV3(), _AlwaysAnswersEngine()
    )
    assert result.verified is False
    assert result.manual_fallback is True


def test_target_proposal_is_not_an_approved_answer_and_has_auditable_state() -> None:
    names = {field.name for field in fields(VqaResult)}
    assert {"proposal", "approved", "verifier_status", "retry_count", "manual_required", "status"} <= names


def test_target_retry_is_bounded_to_one_and_exhaustion_is_manual() -> None:
    engine = _RejectingEngine()
    result = answer_query(
        _candidate(), "What color is the cup?", _CandidateFrameTV1(), _RecordingTV3(), engine,
        max_controlled_retries=1,
    )
    assert engine.answer_calls == 2
    assert result.retry_count == 1
    assert result.manual_fallback is True


def test_target_unavailable_wp04_fails_closed_without_fabricated_answer() -> None:
    result = answer_query(
        _candidate(), "What color is the cup?", _FramesOnlyTV1(), _UnavailableTV3(), _AlwaysAnswersEngine()
    )
    assert result.answer == ""
    assert result.verified is False
    assert result.manual_fallback is True
    assert result.status == "degraded"


def test_target_fixture_response_is_rich_and_not_auto_approved() -> None:
    from tv4.fixtures import VQA_RESPONSE

    result = VQA_RESPONSE["results"][0]
    evidence = result["evidence"]
    assert result["approved"] is False
    assert result["manual_required"] is True
    assert {"query_text", "question", "selected_frames", "asr_evidence", "object_evidence"} <= set(evidence)
