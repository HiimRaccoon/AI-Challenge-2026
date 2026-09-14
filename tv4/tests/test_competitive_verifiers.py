from __future__ import annotations

from tv4.contracts import AsrEvidence, EvidencePack, OcrEvidence, SearchCandidate
from tv4.qa_verifier import verify_answer
from tv4.trake_event_parser import parse_event_graph
from tv4.trake_pipeline import rerun_trake_event
from tv4.trake_verifier import verify_sequence


def _pack(*, ocr: str = "", asr: str = "") -> EvidencePack:
    return EvidencePack(
        query_id="qa-verify",
        video_id="V1",
        frame_id=10,
        timestamp_ms=1_000,
        keyframe_path=None,
        ocr_evidence=(
            OcrEvidence(
                detection_id="ocr-1",
                video_id="V1",
                frame_id=10,
                timestamp_ms=1_000,
                raw_text=ocr,
                normalized_text=ocr.casefold(),
                bbox_xyxy_norm=(0.1, 0.1, 0.8, 0.3),
            ),
        ) if ocr else (),
        asr_evidence=(
            AsrEvidence(
                segment_id="asr-1",
                video_id="V1",
                start_ms=900,
                end_ms=1_500,
                text=asr,
            ),
        ) if asr else (),
    )


def _candidate(frame_id: int, timestamp_ms: int, score: float = 0.2) -> SearchCandidate:
    return SearchCandidate(
        query_id="trake-verify",
        video_id="V1",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        source="fusion",
        rank=1,
        score=score,
        provenance_sources=("visual", "object"),
    )


def test_qa_verifier_independently_supports_answer_bearing_ocr():
    verification = verify_answer(
        "Tên xã là gì?", "Giang Ly", _pack(ocr="UBND XÃ GIANG LY"),
        producer_supported=False,
    )

    assert verification.supported is True
    assert verification.ocr_support == 1.0
    assert verification.final_support_score >= 0.55


def test_qa_verifier_supports_answer_in_structured_asr_context():
    evidence = EvidencePack(
        query_id="qa-context",
        video_id="V1",
        frame_id=10,
        timestamp_ms=1_000,
        keyframe_path=None,
        asr_evidence=(AsrEvidence(
            segment_id="asr-context",
            video_id="V1",
            start_ms=900,
            end_ms=1_500,
            text="Tô cháo đã hoàn chỉnh",
            context=({"start_ms": 1_500, "end_ms": 2_500, "text": "Chà bông cá hồi"},),
        ),),
    )

    verification = verify_answer(
        "Topping là thịt con gì?", "Cá hồi", evidence, producer_supported=True,
    )

    assert verification.audio_text_support == 1.0
    assert verification.supported is True


def test_qa_verifier_flags_explicit_yes_no_contradiction():
    verification = verify_answer(
        "Người này có trả tiền không?", "Có", _pack(asr="Không, tôi chưa trả tiền"),
        producer_supported=True,
    )

    assert verification.contradiction is True
    assert verification.supported is False
    assert verification.manual_required is True


def test_event_parser_tracks_pronoun_and_state_transition():
    graph = parse_event_graph((
        "A man enters the shop",
        "He picks up a product",
        "He gives the product to an employee",
    ))

    assert graph[1].refers_to_previous_entity is True
    assert graph[1].state_transition == "holding"
    assert graph[2].state_transition == "released"


def test_trake_verifier_penalizes_implausible_gap_without_changing_frames():
    events = ("A man enters the shop", "He picks up a product")
    short = (_candidate(10, 1_000), _candidate(20, 20_000))
    long = (_candidate(10, 1_000), _candidate(20, 1_201_000))

    short_verification = verify_sequence(events, short)
    long_verification = verify_sequence(events, long)

    assert short_verification.order_consistent is True
    assert long_verification.implausible_gap_penalty > short_verification.implausible_gap_penalty
    assert short_verification.final_sequence_score > long_verification.final_sequence_score
    assert [item.frame_id for item in short] == [10, 20]


def test_trake_verifier_rejects_non_monotonic_sequence():
    verification = verify_sequence(
        ("event one", "event two"),
        (_candidate(20, 2_000), _candidate(10, 1_000)),
    )

    assert verification.order_consistent is False
    assert verification.contradiction_penalty == 1.0


def test_action_clip_support_requires_upstream_temporal_window():
    from dataclasses import replace

    event = ("He picks up a product",)
    without_window = (_candidate(10, 10_000),)
    with_window = (
        replace(without_window[0], window_start_ms=9_000, window_end_ms=11_000),
    )

    unsupported = verify_sequence(event, without_window)
    available = verify_sequence(event, with_window)

    assert unsupported.clip_manual_required is True
    assert unsupported.micro_clip_support == 0.2
    assert available.clip_manual_required is True
    assert available.micro_clip_support == 0.25


class _RerunVisual:
    def search(self, query_id, query_text, event_index=None):
        return [
            _candidate(15, 15_000, 0.9),
            _candidate(35, 35_000, 0.8),
        ]


class _RerunTv3:
    def search_all(self, request, routes):
        return {
            "asr": [_candidate(25, 25_000, 0.95)],
            "ocr": [_candidate(5, 5_000, 0.9)],
            "object": [],
        }


def test_rerun_event_respects_locked_bounds_and_preserves_locked_inputs():
    from tv4.kis_pipeline import KisServices

    services = KisServices(
        tv1=None,
        tv3=_RerunTv3(),
        visual=_RerunVisual(),
        preprocess_run_id="run-1",
    )
    locked = ((0, 10_000), (2, 30_000))
    alternatives, lower, upper = rerun_trake_event(
        "person enters, speaks, then leaves",
        ("enters", "speaks", "leaves"),
        1,
        "V1",
        services,
        locked_events=locked,
        exclude_frame_id=15,
    )

    assert (lower, upper) == (10_000, 30_000)
    assert [candidate.frame_id for candidate in alternatives] == [25]
    assert locked == ((0, 10_000), (2, 30_000))
    assert alternatives[0].event_index == 1
    assert alternatives[0].preprocess_run_id == "run-1"
