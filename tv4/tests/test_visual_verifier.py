from __future__ import annotations

import base64
import json
from dataclasses import replace

from tv4.contracts import CanonicalFrameReference, EvidencePack, SearchCandidate
from tv4.qa_query_plan import build_qa_query_plan
from tv4.qa_verifier import verify_answer
from tv4.trake_verifier import verify_sequence
from tv4.visual_verifier import (
    OllamaVisualVerifier,
    VisualFrame,
    VisualVerification,
    canonical_qa_frames,
    canonical_trake_frame_groups,
)


def _candidate(frame_id: int = 20, timestamp_ms: int = 2_000) -> SearchCandidate:
    return SearchCandidate(
        query_id="visual-check",
        video_id="V1",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        source="fusion",
        rank=1,
        score=0.2,
        provenance_sources=("visual",),
    )


def test_ollama_visual_verifier_sends_bounded_images_and_parses_strict_json(tmp_path):
    first = tmp_path / "10.jpg"
    second = tmp_path / "20.jpg"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    captured = {}

    def transport(url, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {
            "message": {
                "content": json.dumps({
                    "score": 0.83,
                    "contradiction": False,
                    "rationale": "The action is visible.",
                }),
            },
        }

    verifier = OllamaVisualVerifier(
        enabled=True,
        model="test-vlm",
        max_images=1,
        transport=transport,
    )
    result = verifier.verify_qa(
        "What is held?",
        "a cup",
        (
            VisualFrame("V1", 10, 1_000, first),
            VisualFrame("V1", 20, 2_000, second),
        ),
    )

    assert result.available is True
    assert result.score == 0.83
    assert result.frame_ids == (10,)
    assert captured["url"].endswith("/api/chat")
    images = captured["payload"]["messages"][0]["images"]
    assert len(images) == 1
    assert base64.b64decode(images[0]) == b"first-image"


def test_visual_verifier_fails_closed_on_malformed_model_output(tmp_path):
    image = tmp_path / "10.jpg"
    image.write_bytes(b"image")
    verifier = OllamaVisualVerifier(
        enabled=True,
        model="test-vlm",
        transport=lambda *_: {"message": {"content": "not-json"}},
    )

    result = verifier.verify_qa(
        "question", "answer", (VisualFrame("V1", 10, 1_000, image),),
    )

    assert result.available is False
    assert result.score is None
    assert result.reason == "ValueError"
    assert result.frame_ids == (10,)


def test_visual_chart_answer_is_bounded_and_grounded(tmp_path):
    image = tmp_path / "chart.jpg"
    image.write_bytes(b"jpeg")

    def transport(url, payload, timeout):
        return {"message": {"content": json.dumps({
            "answer": "37.5", "observed_values": [], "score": 0.91,
        })}}

    verifier = OllamaVisualVerifier(enabled=True, model="vision", transport=transport)
    plan = build_qa_query_plan(
        "Biểu đồ sinh trưởng của thực vật ven biển",
        "Loài II tốt nhất ở độ mặn bao nhiêu?",
    )
    result = verifier.answer_qa(
        "Loài II tốt nhất ở độ mặn bao nhiêu?",
        plan,
        (VisualFrame("V1", 10, 1_000, image),),
    )

    assert result.available is True
    assert result.answer == "37.5"
    assert result.frame_ids == (10,)


def test_visual_chart_answer_fails_closed_when_independent_reads_disagree(tmp_path):
    image = tmp_path / "chart-disagreement.jpg"
    image.write_bytes(b"jpeg")
    answers = iter(("18", "24"))

    def transport(url, payload, timeout):
        return {"message": {"content": json.dumps({
            "answer": next(answers), "observed_values": [], "score": 0.95,
        })}}

    verifier = OllamaVisualVerifier(enabled=True, model="vision", transport=transport)
    plan = build_qa_query_plan(
        "Biểu đồ sinh trưởng của thực vật ven biển",
        "Loài II tốt nhất ở độ mặn bao nhiêu?",
    )
    result = verifier.answer_qa(
        "Loài II tốt nhất ở độ mặn bao nhiêu?",
        plan,
        (VisualFrame("V1", 10, 1_000, image),),
    )

    assert result.available is False
    assert result.reason == "chart_reading_not_reproducible"


def test_temporal_answer_requires_exact_set_difference_proof(tmp_path):
    image = tmp_path / "sequence.jpg"
    image.write_bytes(b"jpeg")
    observed = [str(value) for value in (1, 2, 3, 5, 6, 7, 8)]

    def transport(url, payload, timeout):
        return {"message": {"content": json.dumps({
            "answer": "4", "observed_values": observed, "score": 0.88,
        })}}

    verifier = OllamaVisualVerifier(enabled=True, model="vision", transport=transport)
    plan = build_qa_query_plan(
        "Các cột có số",
        "Trong 16 giây đầu tiên số nào không nhìn thấy trong các số từ 1-8?",
    )
    result = verifier.answer_qa(
        "Trong 16 giây đầu tiên số nào không nhìn thấy trong các số từ 1-8?",
        plan,
        (VisualFrame("V1", 1, 0, image),),
    )

    assert result.available is True
    assert result.answer == "4"
    assert result.observed_values == tuple(observed)


def test_visual_verifier_uses_safe_defaults_for_malformed_environment(monkeypatch):
    monkeypatch.setenv("TV4_VISUAL_VERIFIER_ENABLED", "1")
    monkeypatch.setenv("TV4_VISUAL_VERIFIER_MODEL", "test-vlm")
    monkeypatch.setenv("TV4_VISUAL_VERIFIER_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setenv("TV4_VISUAL_VERIFIER_MAX_IMAGES", "-3")

    verifier = OllamaVisualVerifier.from_environment()

    assert verifier.enabled is True
    assert verifier.timeout_seconds == 45.0
    assert verifier.max_images == 5


def test_canonical_qa_frames_only_resolves_tv1_owned_paths_inside_root(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    safe = root / "10.jpg"
    safe.write_bytes(b"safe")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    evidence = EvidencePack(
        query_id="q",
        video_id="V1",
        frame_id=10,
        timestamp_ms=1_000,
        keyframe_path="10.jpg",
        selected_frames=(
            CanonicalFrameReference("V1", 10, 1_000, "10.jpg", "run-1"),
            CanonicalFrameReference("V1", 11, 1_100, str(outside), "run-1"),
            CanonicalFrameReference("V1", 12, 1_200, "missing.jpg", "run-1"),
            CanonicalFrameReference("V2", 13, 1_300, "10.jpg", "run-1"),
        ),
    )

    frames = canonical_qa_frames(
        evidence, root, (".jpg",), preprocess_run_id="run-1",
    )

    assert [(frame.video_id, frame.frame_id, frame.path) for frame in frames] == [
        ("V1", 10, safe),
    ]


class _TV1:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def frames(self, video_id):
        self.calls.append(video_id)
        return self.records


def test_trake_visual_groups_use_only_upstream_ids_in_temporal_order(tmp_path):
    for frame_id in (10, 20, 30):
        (tmp_path / f"{frame_id}.jpg").write_bytes(str(frame_id).encode())
    tv1 = _TV1([
        {"video_id": "V1", "frame_id": 30, "timestamp_ms": 3_000, "keyframe_path": "30.jpg", "preprocess_run_id": "run-1"},
        {"video_id": "V1", "frame_id": 10, "timestamp_ms": 1_000, "keyframe_path": "10.jpg", "preprocess_run_id": "run-1"},
        {"video_id": "V1", "frame_id": 20, "timestamp_ms": 2_000, "keyframe_path": "20.jpg", "preprocess_run_id": "run-1"},
    ])
    candidates = (
        replace(_candidate(), window_start_ms=900, window_end_ms=3_100),
        replace(_candidate(30, 3_000), window_start_ms=2_500, window_end_ms=3_100),
    )

    groups = canonical_trake_frame_groups(
        candidates,
        tv1,
        tmp_path,
        (".jpg",),
        preprocess_run_id="run-1",
        limit_per_event=3,
    )

    assert [[frame.frame_id for frame in group] for group in groups] == [[10, 20, 30], [30]]
    assert tv1.calls == ["V1"]


def test_visual_checks_feed_qa_and_action_verifiers_without_changing_candidates():
    evidence = EvidencePack(
        query_id="q", video_id="V1", frame_id=20, timestamp_ms=2_000,
        keyframe_path=None,
    )
    visual = VisualVerification(
        available=True,
        score=0.9,
        provider="ollama-vlm",
        model="test-vlm",
        frame_ids=(10, 20, 30),
    )
    qa = verify_answer(
        "What is held?", "a cup", evidence, visual_verification=visual,
    )
    candidate = _candidate()
    trake = verify_sequence(
        ("He picks up a cup",),
        (candidate,),
        visual_verifications=(visual,),
    )

    assert qa.supported is True
    assert qa.visual_support == 0.9
    assert qa.visual_frame_ids == (10, 20, 30)
    assert trake.micro_clip_support == 0.9
    assert trake.clip_manual_required is False
    assert trake.visual_verifier_available is True
    assert candidate.frame_id == 20
