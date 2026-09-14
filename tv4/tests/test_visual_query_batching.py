from types import SimpleNamespace

from tv4.contracts import SearchCandidate
from tv4.kis_pipeline import _blend_visual_query_results, _search_visual_query_set_with_scope, run_kis_query


class _BatchOnlyVisual:
    def __init__(self):
        self.calls = []

    def search(self, *args, **kwargs):
        raise AssertionError("independent search should not run")

    def search_batch(self, requests):
        self.calls.append(requests)
        return [
            [SearchCandidate(
                query_id=query_id, video_id="L01_V001", frame_id=index + 1,
                timestamp_ms=(index + 1) * 10_000, source="visual", rank=1, score=1.0,
            )]
            for index, (query_id, _, _) in enumerate(requests)
        ]


class _ModelAwareBatchVisual(_BatchOnlyVisual):
    temporal_models = ("bge_vl", "metaclip2")
    multi_moment_models = ("beit3", "metaclip2")

    def search_batch(self, requests, **kwargs):
        self.calls.append((requests, kwargs))
        return [
            [SearchCandidate(
                query_id=query_id, video_id="L01_V001", frame_id=index + 1,
                timestamp_ms=(index + 1) * 10_000, source="visual", rank=1, score=1.0,
            )]
            for index, (query_id, _, _) in enumerate(requests)
        ]


class _KisTv3WithoutTranscriptScope:
    def __init__(self):
        self.transcript_scope_calls = 0

    def search_transcript_windows(self, **kwargs):
        self.transcript_scope_calls += 1
        return []

    def search_all(self, request, routes):
        return {}


class _OneResultVisual:
    top_k = 100
    candidate_k_per_model = 100

    def search(self, query_id, query_text, **kwargs):
        return [SearchCandidate(query_id, "VIDEO", 10, 1_000, "visual", rank=1)]


class _OneFrameAuthority:
    def frames(self, video_id):
        return [{
            "video_id": video_id, "frame_id": 10, "timestamp_ms": 1_000,
            "preprocess_run_id": "run", "keyframe_path": "keyframes/VIDEO/0.jpg",
        }]


class _LocatorTv3:
    def __init__(self):
        self.queries = []

    def search(self, route, request):
        self.queries.append((route, request.query_text))
        return [SearchCandidate(
            request.query_id, "L25_V062", 13466, 538_640, "asr", rank=1,
        )]

    def search_all(self, request, routes):
        return {}


class _ScopedVisual:
    top_k = 100
    candidate_k_per_model = 100
    multi_moment_models = ("beit3",)

    def __init__(self):
        self.scope = []

    def search_batch(self, requests, **kwargs):
        self.scope = kwargs.get("candidate_frames", [])
        return [[SearchCandidate(
            query_id, "L25_V062", 13718, 548_720, "visual", rank=1,
        )] for query_id, _, _ in requests]


class _LocatorFrames:
    def frames(self, video_id):
        return [
            {
                "video_id": video_id, "frame_id": frame_id,
                "timestamp_ms": timestamp_ms, "preprocess_run_id": "run",
                "keyframe_path": f"keyframes/{video_id}/{frame_id}.jpg",
            }
            for frame_id, timestamp_ms in (
                (13466, 538_640), (13718, 548_720), (14022, 560_880),
            )
        ]


def test_multiple_visual_variants_share_one_gpu_batch():
    visual = _BatchOnlyVisual()
    services = SimpleNamespace(visual=visual)

    result = _search_visual_query_set_with_scope(
        services, "q", ("first visible moment", "second visible moment"), [],
    )

    assert len(visual.calls) == 1
    assert len(visual.calls[0]) == 2
    assert {candidate.frame_id for candidate in result} == {1, 2}


def test_kis_visual_batch_selects_models_by_query_structure():
    visual = _ModelAwareBatchVisual()
    services = SimpleNamespace(visual=visual)

    _search_visual_query_set_with_scope(
        services, "ordered", ("anchor", "locator", "first", "then"), [],
        temporal_chain=True,
    )
    _search_visual_query_set_with_scope(
        services, "unordered", ("anchor", "locator", "object one", "object two"), [],
    )

    assert visual.calls[0][1]["models"] == ("bge_vl", "metaclip2")
    assert visual.calls[1][1]["models"] == ("beit3", "metaclip2")


def test_kis_does_not_serialize_optional_transcript_scope_before_visual_search():
    tv3 = _KisTv3WithoutTranscriptScope()
    services = SimpleNamespace(
        visual=_OneResultVisual(), tv3=tv3,
        tv1=_OneFrameAuthority(), preprocess_run_id="run", refine=None,
        derivative_asset_root=None,
    )

    ranked = run_kis_query(
        "A lecture slide with visible figures", services,
        query_id="q", visual_query_texts=("lecture slide figures",),
    )

    assert ranked[0].video_id == "VIDEO"
    assert tv3.transcript_scope_calls == 0


def test_symbolic_lecture_query_uses_asr_anchor_to_scope_exact_visual_frame():
    tv3 = _LocatorTv3()
    visual = _ScopedVisual()
    services = SimpleNamespace(
        visual=visual, tv3=tv3, tv1=_LocatorFrames(),
        preprocess_run_id="run", refine=None, derivative_asset_root=None,
    )
    query = (
        "Đây là một đoạn trong bài giảng.\n"
        "- Một nhóm nhân vật 3D màu trắng vây quanh nhân vật đỏ.\n"
        "- Hai nhân vật hoạt hình đang thi đấu kéo co."
    )

    ranked = run_kis_query(
        query, services, query_id="q",
        visual_query_texts=("lecture slide figures", "cartoon tug of war"),
    )

    assert tv3.queries == [("asr", "cạnh tranh ganh đua đấu tranh")]
    assert {frame["frame_id"] for frame in visual.scope} == {13466, 13718, 14022}
    assert ranked[0].video_id == "L25_V062"
    assert ranked[0].frame_id == 13718


def test_vqa_supplemental_query_cannot_reorder_primary_anchor():
    primary = [
        SearchCandidate(
            query_id="q", video_id="CORRECT", frame_id=1,
            timestamp_ms=1_000, source="visual", rank=1,
        ),
        SearchCandidate(
            query_id="q", video_id="GENERIC", frame_id=2,
            timestamp_ms=2_000, source="visual", rank=2,
        ),
    ]
    supplemental = [SearchCandidate(
        query_id="q", video_id="GENERIC", frame_id=2,
        timestamp_ms=2_000, source="visual", rank=1,
    )]

    ranked = _blend_visual_query_results(
        [primary, supplemental], preserve_primary_order=True,
    )

    assert ranked[0].video_id == "CORRECT"


def test_multi_moment_video_consensus_promotes_cross_scene_match():
    first = [
        SearchCandidate("q", "LOOKALIKE", 1, 1_000, "visual", rank=1),
        SearchCandidate("q", "COMPLETE", 2, 2_000, "visual", rank=40),
    ]
    second = [
        SearchCandidate("q", "OTHER", 3, 3_000, "visual", rank=1),
        SearchCandidate("q", "COMPLETE", 4, 20_000, "visual", rank=40),
    ]
    third = [
        SearchCandidate("q", "THIRD", 5, 4_000, "visual", rank=1),
        SearchCandidate("q", "COMPLETE", 6, 40_000, "visual", rank=40),
    ]

    ranked = _blend_visual_query_results(
        [first, second, third], video_consensus_weight=0.85,
    )

    assert ranked[0].video_id == "COMPLETE"
    assert ranked[0].provenance["visual_video_support_count"] == 3


def test_temporal_chain_returns_first_frame_of_compact_ordered_sequence():
    anchor = [
        SearchCandidate("q", "CORRECT", 722, 28_880, "visual", rank=8),
        SearchCandidate("q", "LATE", 7080, 283_200, "visual", rank=1),
    ]
    first_moment = [
        SearchCandidate("q", "LATE", 7080, 283_200, "visual", rank=1),
        SearchCandidate("q", "CORRECT", 580, 23_200, "visual", rank=8),
    ]
    second_moment = [
        SearchCandidate("q", "LATE", 900, 36_000, "visual", rank=1),
        SearchCandidate("q", "CORRECT", 722, 28_880, "visual", rank=8),
    ]
    final_moment = [
        SearchCandidate("q", "LATE", 1100, 44_000, "visual", rank=1),
        SearchCandidate("q", "CORRECT", 816, 32_640, "visual", rank=8),
    ]

    ranked = _blend_visual_query_results(
        [anchor, first_moment, second_moment, final_moment],
        temporal_chain=True,
        temporal_max_span_ms=60_000,
    )

    assert ranked[0].video_id == "CORRECT"
    assert ranked[0].frame_id == 580
    assert ranked[0].timestamp_ms == 23_200
    assert ranked[0].provenance["visual_temporal_chain_valid"] is True
    assert [step["frame_id"] for step in ranked[0].provenance["visual_temporal_chain"]] == [580, 722, 816]
    assert ranked[0].provenance["visual_temporal_chain_span_ms"] == 9_440


def test_temporal_chain_falls_back_when_no_video_covers_every_moment():
    primary = [SearchCandidate("q", "ANCHOR", 1, 1_000, "visual", rank=1)]
    first = [SearchCandidate("q", "FIRST_ONLY", 2, 2_000, "visual", rank=1)]
    second = [SearchCandidate("q", "SECOND_ONLY", 3, 3_000, "visual", rank=1)]

    ranked = _blend_visual_query_results(
        [primary, first, second], temporal_chain=True,
    )

    assert ranked
    assert all(not item.provenance.get("visual_temporal_chain_valid", False) for item in ranked)


def test_temporal_cluster_localizes_when_one_moment_has_no_recall():
    anchor = [SearchCandidate("q", "LATE", 7080, 283_200, "visual", rank=1)]
    locator = [
        SearchCandidate("q", "CORRECT", 722, 28_880, "visual", rank=1),
        SearchCandidate("q", "LATE", 7080, 283_200, "visual", rank=30),
    ]
    first_moment = [
        SearchCandidate("q", "CORRECT", 792, 31_680, "visual", rank=10),
        SearchCandidate("q", "LATE", 7096, 283_840, "visual", rank=5),
    ]
    second_moment = [
        SearchCandidate("q", "CORRECT", 744, 29_760, "visual", rank=1),
    ]
    missing_final_moment = [
        SearchCandidate("q", "OTHER", 50, 2_000, "visual", rank=1),
    ]

    ranked = _blend_visual_query_results(
        [anchor, locator, first_moment, second_moment, missing_final_moment],
        temporal_chain=True,
        temporal_moment_start_index=2,
    )

    assert ranked[0].video_id == "CORRECT"
    assert 28_000 <= ranked[0].timestamp_ms <= 32_000
    assert ranked[0].provenance["visual_temporal_cluster_valid"] is True
    assert ranked[0].provenance["visual_temporal_cluster_moment_support"] == 2


def test_strong_partial_cluster_outranks_weak_deep_tail_chain():
    anchor = [
        SearchCandidate("q", "WEAK_CHAIN", 1, 1_000, "visual", rank=1),
    ]
    locator = [
        SearchCandidate("q", "CORRECT", 10, 10_000, "visual", rank=1),
        SearchCandidate("q", "WEAK_CHAIN", 1, 1_000, "visual", rank=1_000),
    ]
    first_moment = [
        SearchCandidate("q", "CORRECT", 11, 11_000, "visual", rank=1),
        SearchCandidate("q", "WEAK_CHAIN", 1, 1_000, "visual", rank=300),
    ]
    second_moment = [
        SearchCandidate("q", "CORRECT", 12, 12_000, "visual", rank=1),
        SearchCandidate("q", "WEAK_CHAIN", 2, 2_000, "visual", rank=300),
    ]
    final_moment = [
        SearchCandidate("q", "OTHER", 20, 20_000, "visual", rank=1),
        SearchCandidate("q", "WEAK_CHAIN", 3, 3_000, "visual", rank=300),
    ]

    ranked = _blend_visual_query_results(
        [anchor, locator, first_moment, second_moment, final_moment],
        temporal_chain=True,
        temporal_moment_start_index=2,
    )

    assert ranked[0].video_id == "CORRECT"
    assert ranked[0].provenance["visual_temporal_cluster_valid"] is True
    weak = next(item for item in ranked if item.video_id == "WEAK_CHAIN")
    assert weak.provenance["visual_temporal_chain_valid"] is True
