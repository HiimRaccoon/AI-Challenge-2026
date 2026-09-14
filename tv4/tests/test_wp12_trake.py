from tv4.contracts import SearchCandidate
from tv4.trake_pipeline import visual_query_for_trake
from tv4.wp12_trake import EventPool, align_dp, align_greedy, align_trake


def test_lion_dance_visual_expansion_preserves_action_cues():
    visual_query = visual_query_for_trake(
        "M\u00faa l\u00e2n m\u00e0u v\u00e0ng \u0111en tr\u1eafng quay v\u00f2ng tr\u00ean c\u1ed9t, r\u1ed3i ch\u00e0o r\u1ed3ng"
    )
    assert "lion dance" in visual_query
    assert "performance" in visual_query


def test_lion_dance_visual_expansion_applies_sequence_context_to_short_event():
    visual_query = visual_query_for_trake(
        "Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.",
        context="Đoạn video múa lân một con lân màu vàng đen trắng",
    )
    for word in ("lion dance", "landing"):
        assert word in visual_query


def test_cooking_visual_expansion_keeps_subject_for_short_event():
    visual_query = visual_query_for_trake(
        "Khoảnh khắc đầu tiên cắt củ năng.",
        context="Trong đoạn video nấu ăn một món ăn về nấm, gồm các khoảnh khắc sơ chế",
    )
    for word in ("cooking", "mushrooms", "cutting", "water chestnuts"):
        assert word in visual_query


def _cand(frame_id, ts, score):
    return SearchCandidate(
        query_id="q1", video_id="L10_V010", frame_id=frame_id, timestamp_ms=ts,
        source="fusion", rank=1, score=score,
    )


def test_align_dp_prefers_monotonic_path():
    pools = [
        EventPool(0, (_cand(1, 100, 0.9), _cand(2, 300, 0.5))),
        EventPool(1, (_cand(3, 150, 0.9), _cand(4, 400, 0.4))),  # frame 3 (ts=150) is best but breaks monotonicity with frame 1 only if event0 picks 2
    ]
    result = align_dp(pools)
    assert result is not None
    assert result[0].timestamp_ms <= result[1].timestamp_ms


def test_align_dp_returns_none_when_no_monotonic_path_exists():
    pools = [
        EventPool(0, (_cand(1, 900, 0.9),)),
        EventPool(1, (_cand(2, 100, 0.9),)),
    ]
    assert align_dp(pools) is None


def test_align_trake_rejects_a_sequence_with_no_forward_path():
    pools = [
        EventPool(0, (_cand(1, 900, 0.9),)),
        EventPool(1, (_cand(2, 100, 0.9),)),
    ]
    result = align_trake(pools, strategy="dp")
    assert result is None


def test_align_trake_missing_event_pool_is_none():
    pools = [EventPool(0, ()), EventPool(1, (_cand(2, 100, 0.9),))]
    assert align_trake(pools) is None


def test_align_trake_does_not_reuse_the_same_frame_for_later_event():
    pools = [
        EventPool(0, (_cand(1, 100, 0.9),)),
        EventPool(1, (_cand(1, 100, 0.8),)),
    ]
    assert align_trake(pools) is None


class MockVisualBatchClient:
    def __init__(self):
        self.batch_calls = []

    def search_batch(self, requests):
        self.batch_calls.append(requests)
        # requests: [(qid, qtxt, ev_idx)]
        results = []
        for qid, qtxt, ev_idx in requests:
            if ev_idx is None:
                # Whole query
                results.append([_cand(10, 100, 0.95), _cand(20, 200, 0.9)])
            elif ev_idx == 0:
                results.append([_cand(10, 100, 0.9), _cand(15, 150, 0.5)])
            elif ev_idx == 1:
                results.append([_cand(20, 200, 0.9), _cand(10, 100, 0.3)])
        return results


class MockTv3Client:
    def __init__(self):
        self.calls = []

    def search_all(self, req, routes):
        self.calls.append((req, tuple(routes)))
        return {}


class _NoEventCoverageVisual:
    def search_batch(self, requests):
        return [
            [_cand(1, 100, 0.9)],  # whole query: L10_V010
            [],                    # event 1 has no visual evidence in it
            [_cand(2, 200, 0.9)],
        ]


def _video_cand(video_id, frame_id, ts, score):
    return SearchCandidate(
        query_id="q1", video_id=video_id, frame_id=frame_id, timestamp_ms=ts,
        source="visual", rank=1, score=score,
    )


class _EventOnlyVideoVisual:
    def search_batch(self, requests):
        return [
            [_video_cand("L10_V010", 1, 100, 0.99)],
            [_video_cand("L20_V020", 10, 1_000, 0.80)],
            [_video_cand("L20_V020", 20, 2_000, 0.79)],
        ]


class _CanonicalFramesTv1:
    def frames(self, video_id):
        return [
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "keyframe_path": f"{frame_id}.jpg",
                "preprocess_run_id": "run_v1_batch1",
            }
            for frame_id, timestamp_ms in ((10, 100), (15, 150), (20, 200))
        ]


class _VisualVerifier:
    enabled = True
    max_sequence_candidates = 3
    max_images = 5

    def verify_events(self, events, frame_groups):
        from tv4.visual_verifier import VisualVerification

        return tuple(
            VisualVerification(
                available=True,
                score=0.9,
                provider="test-vlm",
                frame_ids=tuple(frame.frame_id for frame in frames),
            )
            for frames in frame_groups
        )


def test_run_trake_query_applies_late_visual_verifier_to_canonical_tv1_frames(tmp_path):
    from tv4.kis_pipeline import KisServices
    from tv4.trake_pipeline import run_trake_query

    for frame_id in (10, 15, 20):
        (tmp_path / f"{frame_id}.jpg").write_bytes(b"image")
    services = KisServices(
        tv1=_CanonicalFramesTv1(),
        tv3=MockTv3Client(),
        visual=MockVisualBatchClient(),
        refine=None,
        preprocess_run_id="run_v1_batch1",
        derivative_asset_root=tmp_path,
        allowed_image_extensions=frozenset({".jpg"}),
    )

    hypothesis = run_trake_query(
        "high jump",
        services,
        events=["athlete picks up a cup", "athlete gives the cup away"],
        query_id="trake-visual-verify",
        visual_verifier=_VisualVerifier(),
    )

    assert hypothesis is not None
    assert hypothesis.frame_ids == (10, 20)
    assert hypothesis.verification["visual_verifier_available"] is True
    assert hypothesis.verification["visual_frame_ids"] == [[10, 15, 20], [10, 15, 20]]
    assert hypothesis.verification["micro_clip_support"] == 0.9


def test_run_trake_query_invokes_visual_search_batch(monkeypatch):
    monkeypatch.setenv("TV4_EXTERNAL_ASSIST_ENABLED", "0")
    from tv4.kis_pipeline import KisServices
    from tv4.trake_pipeline import run_trake_query

    mock_visual = MockVisualBatchClient()
    mock_tv3 = MockTv3Client()
    services = KisServices(
        tv1=None,
        tv3=mock_tv3,
        visual=mock_visual,
        refine=None,
        preprocess_run_id="run_v1_batch1",
    )

    hyp = run_trake_query(
        "Vận động viên thực hiện cú nhảy cao",
        services,
        events=["Giậm nhảy", "Bay qua xà"],
        query_id="trake-test-01",
    )

    assert hyp is not None
    assert hyp.verification["order_consistent"] is True
    assert 0.0 <= hyp.verification["final_sequence_score"] <= 1.0
    assert hyp.video_id == "L10_V010"
    assert len(hyp.frame_ids) == 2
    # Ensure visual search_batch was called exactly ONCE with 3 requests (whole + 2 events)
    assert len(mock_visual.batch_calls) == 1
    assert len(mock_visual.batch_calls[0]) == 3
    assert len(mock_tv3.calls) == 2
    tv3_calls = sorted(mock_tv3.calls, key=lambda call: call[0].event_index)
    assert [call[0].event_index for call in tv3_calls] == [0, 1]
    assert [call[0].query_text for call in tv3_calls] == [
        "Giậm nhảy", "Bay qua xà",
    ]
    assert all(call[1] == ("ocr", "asr", "object") for call in tv3_calls)
    whole_query = "Vận động viên thực hiện cú nhảy cao"
    assert mock_visual.batch_calls[0][0][1] == visual_query_for_trake(whole_query)
    assert mock_visual.batch_calls[0][1][1] == visual_query_for_trake("Giậm nhảy", context=whole_query)
    assert mock_visual.batch_calls[0][2][1] == visual_query_for_trake("Bay qua xà", context=whole_query)


def test_trake_keeps_video_found_only_by_all_event_queries():
    from tv4.kis_pipeline import KisServices
    from tv4.trake_pipeline import run_trake_query

    services = KisServices(
        tv1=None,
        tv3=MockTv3Client(),
        visual=_EventOnlyVideoVisual(),
        refine=None,
        preprocess_run_id="run_v1_batch1",
    )

    hypothesis = run_trake_query(
        "two ordered actions",
        services,
        events=["first action", "second action"],
        query_id="trake-event-only-video",
    )

    assert hypothesis is not None
    assert hypothesis.video_id == "L20_V020"
    assert hypothesis.frame_ids == (10, 20)
