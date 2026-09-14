from tv4.contracts import SearchCandidate
from tv4.external_assist import ExternalVlmVerifier
from tv4.kis_pipeline import KisServices, _external_vlm_rerank


class _ConfidentWrongVerifier:
    enabled = True
    max_images = 10

    def rank(self, query, image_paths):
        return [(1, 100.0)]


def _candidate(frame_id, score):
    return SearchCandidate(
        query_id="q", video_id=f"L01_V{frame_id:03d}", frame_id=frame_id,
        timestamp_ms=frame_id * 1_000, source="fusion", rank=frame_id,
        score=score, evidence_refs=(f"keyframe:{frame_id}.jpg",),
    )


def test_external_vlm_cannot_promote_a_much_weaker_candidate_to_top1(tmp_path, monkeypatch):
    for frame_id in (1, 2):
        (tmp_path / f"{frame_id}.jpg").write_bytes(b"image")
    monkeypatch.setattr(
        ExternalVlmVerifier,
        "from_environment",
        staticmethod(lambda: _ConfidentWrongVerifier()),
    )
    services = KisServices(
        tv1=None, tv3=None, visual=None,
        derivative_asset_root=tmp_path,
        allowed_image_extensions=frozenset({".jpg"}),
    )

    result = _external_vlm_rerank(
        [_candidate(1, 1.0), _candidate(2, 0.5)], "query", services,
    )

    assert [candidate.frame_id for candidate in result] == [1, 2]
    assert result[1].provenance["external_vlm_score"] == 100.0
    assert result[1].provenance["external_vlm_rrf_bonus"] == 0.01
