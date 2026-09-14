"""TV4 must consume only WP09-proven canonical identities."""

from __future__ import annotations

from pathlib import Path
import unittest


class TV4WP09IntegrationTests(unittest.TestCase):
    def test_adapter_has_no_raw_pts_frame_id_fallback(self) -> None:
        source = (Path(__file__).parents[1] / "src" / "tv4" / "adapters" / "wp09_adapter.py").read_text(encoding="utf-8")
        self.assertNotIn("return int(raw_frame.pts)", source)
        self.assertNotIn('frame_id_source: "raw_pts_fallback"', source)

    def test_only_resolver_proven_selectable_result_may_replace_a_kis_anchor(self) -> None:
        from tv4.kis_pipeline import canonical_refined_candidate

        original = {"frame_id": 700, "timestamp_ms": 0}
        raw_pts = {"frame_id": 10_000, "timestamp_ms": 0, "mapping_guaranteed": True, "submission_selectable": True, "identity_source": "raw_pts"}
        fixture = {"frame_id": 701, "timestamp_ms": 41, "mapping_guaranteed": True, "submission_selectable": True, "identity_source": "certified_run_consecutive_original_decode", "provenance_mode": "fixture", "media_identity_verified": True, "producer_compatibility_verified": True, "certification_id": "e4", "certification_report_sha256": "a" * 64}
        proven = {"frame_id": 701, "timestamp_ms": 41, "mapping_guaranteed": True, "submission_selectable": True, "identity_source": "certified_run_consecutive_original_decode", "provenance_mode": "live", "media_identity_verified": True, "producer_compatibility_verified": True, "certification_id": "e4", "certification_report_sha256": "a" * 64}

        self.assertIsNone(canonical_refined_candidate(raw_pts))
        self.assertIsNone(canonical_refined_candidate(fixture))
        self.assertEqual(canonical_refined_candidate(proven), (701, 41))

    def test_narrow_neighbor_api_is_present_for_operator_frame_stepping(self) -> None:
        source = (Path(__file__).parents[1] / "src" / "tv4" / "api.py").read_text(encoding="utf-8")
        self.assertIn('"/exact-frame/neighbors"', source)
