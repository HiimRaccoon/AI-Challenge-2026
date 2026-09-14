from tv4.contracts import SearchCandidate
from tv4.wp10_fusion import fuse_kis, reciprocal_rank_fusion


def _cand(video_id, frame_id, ts, source, rank, score=None):
    return SearchCandidate(
        query_id="q1", video_id=video_id, frame_id=frame_id, timestamp_ms=ts,
        source=source, rank=rank, score=score,
    )


def test_agreeing_branches_outrank_single_branch():
    visual = [_cand("L01_V001", 100, 5000, "visual", 1), _cand("L02_V002", 50, 1000, "visual", 2)]
    ocr = [_cand("L01_V001", 100, 5000, "ocr", 1)]
    fused = fuse_kis({"visual": visual, "ocr": ocr})
    assert fused[0].video_id == "L01_V001"
    assert fused[0].rank == 1
    assert "ocr" in fused[0].provenance_sources and "visual" in fused[0].provenance_sources


def test_empty_branch_does_not_crash():
    fused = fuse_kis({"visual": [], "ocr": [], "asr": []})
    assert fused == []


def test_dedup_window_merges_near_duplicate_frames():
    visual = [_cand("L01_V001", 100, 5000, "visual", 1), _cand("L01_V001", 101, 5050, "visual", 2)]
    fused = reciprocal_rank_fusion({"visual": visual}, dedup_window_ms=1000)
    assert len(fused) == 1


def test_diversity_caps_per_video():
    visual = [_cand("L01_V001", i, i * 5000, "visual", i + 1) for i in range(10)]
    fused = fuse_kis({"visual": visual}, top_k=100)
    same_video = [c for c in fused if c.video_id == "L01_V001"]
    assert len(same_video) <= 5


def test_weighted_rrf_prioritizes_higher_weighted_branch():
    visual = [_cand("L01_V001", 100, 5000, "visual", 1)]
    ocr = [_cand("L02_V002", 200, 6000, "ocr", 1)]
    # With equal weight, both rank 1 have same score
    fused_equal = fuse_kis({"visual": visual, "ocr": ocr}, weights={"visual": 1.0, "ocr": 1.0})
    assert len(fused_equal) == 2
    # With higher OCR weight (e.g. 2.0 vs 0.5), OCR candidate must take Top 1
    fused_ocr_heavy = fuse_kis({"visual": visual, "ocr": ocr}, weights={"visual": 0.5, "ocr": 2.0})
    assert fused_ocr_heavy[0].video_id == "L02_V002"
    # With higher Visual weight (e.g. 2.0 vs 0.2), Visual candidate must take Top 1
    fused_vis_heavy = fuse_kis({"visual": visual, "ocr": ocr}, weights={"visual": 2.0, "ocr": 0.2})
    assert fused_vis_heavy[0].video_id == "L01_V001"


def test_default_fusion_retains_lower_score_video_for_late_verification():
    visual = [_cand("L01_V001", 1, 1_000, "visual", 1)]
    metadata = [_cand("L99_V999", 2, 2_000, "metadata", 100)]

    fused = fuse_kis({"visual": visual, "metadata": metadata}, top_k=100)

    assert [candidate.video_id for candidate in fused] == ["L01_V001", "L99_V999"]


def test_relative_score_pruning_remains_available_as_explicit_policy():
    visual = [_cand("L01_V001", 1, 1_000, "visual", 1)]
    metadata = [_cand("L99_V999", 2, 2_000, "metadata", 100)]

    fused = fuse_kis(
        {"visual": visual, "metadata": metadata}, top_k=100, min_relative_score=0.5,
    )

    assert [candidate.video_id for candidate in fused] == ["L01_V001"]
