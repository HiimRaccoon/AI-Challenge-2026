from types import SimpleNamespace

from tv4.contracts import SearchCandidate
from tv4.kis_pipeline import _canonicalize_branch_results, _vqa_ranked_candidates


class _FrameAuthority:
    def __init__(self):
        self.calls: list[str] = []

    def frames(self, video_id: str):
        self.calls.append(video_id)
        return [
            {
                "video_id": video_id,
                "frame_id": 6044,
                "timestamp_ms": 241_760,
                "preprocess_run_id": "run-active",
                "keyframe_path": f"keyframes/{video_id}/366.jpg",
            },
            {
                "video_id": video_id,
                "frame_id": 6062,
                "timestamp_ms": 242_480,
                "preprocess_run_id": "run-active",
                "keyframe_path": f"keyframes/{video_id}/367.jpg",
            },
        ]


def _candidate(source: str) -> SearchCandidate:
    return SearchCandidate(
        query_id="q", video_id="L23_V005", frame_id=6064,
        timestamp_ms=242_560, source=source, rank=1,
        evidence_refs=(f"{source}:evidence", "keyframe:keyframes/L23_V005/stale.jpg"),
        preprocess_run_id="run-stale",
    )


def test_every_branch_is_mapped_to_one_active_preview_identity():
    tv1 = _FrameAuthority()
    services = SimpleNamespace(tv1=tv1, preprocess_run_id="run-active")

    result = _canonicalize_branch_results(
        {
            "visual": [_candidate("visual")],
            "ocr": [_candidate("ocr")],
            "object": [_candidate("object")],
            "metadata": [_candidate("metadata")],
        },
        services,
    )

    assert tv1.calls == ["L23_V005"]
    for branch, candidates in result.items():
        candidate = candidates[0]
        assert candidate.frame_id == 6062, branch
        assert candidate.timestamp_ms == 242_480, branch
        assert candidate.preprocess_run_id == "run-active", branch
        assert candidate.provenance["source_frame_id"] == 6064
        assert candidate.provenance["frame_identity_mapping"] == "nearest_timestamp"
        assert "keyframe:keyframes/L23_V005/367.jpg" in candidate.evidence_refs
        assert "keyframe:keyframes/L23_V005/stale.jpg" not in candidate.evidence_refs


def test_visual_vqa_description_is_not_buried_by_generic_transcript_hit():
    visual = SearchCandidate(
        query_id="q", video_id="CORRECT", frame_id=10, timestamp_ms=1_000,
        source="visual", rank=1,
    )
    asr = SearchCandidate(
        query_id="q", video_id="WRONG", frame_id=20, timestamp_ms=2_000,
        source="asr", rank=1, raw_score=10.0,
    )
    weak_visual = SearchCandidate(
        query_id="q", video_id="WRONG", frame_id=21, timestamp_ms=2_100,
        source="visual", rank=10,
    )

    ranked = _vqa_ranked_candidates(
        {"visual": [visual, weak_visual], "asr": [asr]},
        question="Con số trên xe là số mấy?",
        description="Một chiếc xe màu trắng đang rẽ trái",
        top_k=10,
    )

    assert ranked[0].video_id == "CORRECT"


def test_spoken_location_question_keeps_asr_first():
    visual = SearchCandidate(
        query_id="q", video_id="VISUAL", frame_id=10, timestamp_ms=1_000,
        source="visual", rank=1,
    )
    asr = SearchCandidate(
        query_id="q", video_id="SPOKEN", frame_id=20, timestamp_ms=2_000,
        source="asr", rank=2, raw_score=10.0,
    )

    ranked = _vqa_ranked_candidates(
        {"visual": [visual], "asr": [asr]},
        question="Quán trọ nằm trên đường nào?",
        description="Một cụ ông đang trò chuyện",
        top_k=10,
    )

    assert ranked[0].video_id == "SPOKEN"
