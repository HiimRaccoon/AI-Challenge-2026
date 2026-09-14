"""E4-1A tests for TV4's exact-frame trust boundary."""

from __future__ import annotations

from pathlib import Path
import unittest

from tv4.contracts import exact_neighbor_response_is_safe
from tv4.media_identity import MediaRecord, resolve_original_media_path


class TV4ExactIdentityTests(unittest.TestCase):
    def test_path_traversal_and_unknown_video_are_rejected(self) -> None:
        root = Path("C:/corpus")
        registry = {"V1": MediaRecord("V1", "V1.mp4", "a" * 64, "run-1", "1/1000", "media/V1", "mapping/V1")}
        with self.assertRaises(ValueError):
            resolve_original_media_path(root, "../secret", registry, "run-1")
        with self.assertRaises(ValueError):
            resolve_original_media_path(root, "V2", registry, "run-1")
        escaped = {"V1": MediaRecord("V1", "../secret.mp4", "a" * 64, "run-1", "1/1000", "media/V1", "mapping/V1")}
        with self.assertRaises(ValueError):
            resolve_original_media_path(root, "V1", escaped, "run-1")
        wrong_run = {"V1": MediaRecord("V1", "V1.mp4", "a" * 64, "old-run", "1/1000", "media/V1", "mapping/V1")}
        with self.assertRaises(ValueError):
            resolve_original_media_path(root, "V1", wrong_run, "run-1")

    def test_fixture_or_malformed_selectable_payload_is_never_live_safe(self) -> None:
        fixture = {"provenance_mode": "fixture", "video_id": "V1", "anchor_frame_id": 1, "steps": []}
        malformed = {"provenance_mode": "live", "video_id": "V1", "anchor_frame_id": 1, "steps": [{"offset": 1, "frame": {"mapping_guaranteed": True, "submission_selectable": True}}]}
        self.assertFalse(exact_neighbor_response_is_safe(fixture, "V1", 1, [1], "run-1"))
        self.assertFalse(exact_neighbor_response_is_safe(malformed, "V1", 1, [1], "run-1"))

    def test_certified_live_proof_requires_certification_identity(self) -> None:
        frame = {"video_id": "V1", "frame_id": 2, "timestamp_ms": 33, "pts": 1033, "time_base": "1/1000",
                 "preprocess_run_id": "run-1", "mapping_guaranteed": True, "submission_selectable": True,
                 "identity_source": "certified_run_consecutive_original_decode", "media_identity_verified": True,
                 "producer_compatibility_verified": True, "certification_id": "e4-1b", "certification_report_sha256": "a" * 64}
        payload = {"provenance_mode": "live", "video_id": "V1", "anchor_frame_id": 1, "steps": [{"offset": 1, "frame": frame}]}
        self.assertTrue(exact_neighbor_response_is_safe(payload, "V1", 1, [1], "run-1"))
        del frame["certification_id"]
        self.assertFalse(exact_neighbor_response_is_safe(payload, "V1", 1, [1], "run-1"))

    def test_trusted_authority_rejects_forged_live_looking_proof(self) -> None:
        frame = {"video_id": "V1", "frame_id": 2, "timestamp_ms": 33, "pts": 1033, "time_base": "1/1000",
                 "preprocess_run_id": "run-1", "mapping_guaranteed": True, "submission_selectable": True,
                 "identity_source": "certified_run_consecutive_original_decode", "media_identity_verified": True,
                 "producer_compatibility_verified": True, "certification_id": "forged", "certification_report_sha256": "a" * 64,
                 "source_sha256": "b" * 64}
        payload = {"provenance_mode": "live", "video_id": "V1", "anchor_frame_id": 1, "steps": [{"offset": 1, "frame": frame}]}
        self.assertFalse(exact_neighbor_response_is_safe(payload, "V1", 1, [1], "run-1", certification_id="real", certification_report_sha256="c" * 64, source_sha256="b" * 64, time_base="1/1000"))

    def test_repeated_step_preserves_certified_anchor_and_uses_cumulative_offset(self) -> None:
        from tv4.exact_identity import certified_step_request

        second_step = certified_step_request(701, 33, [1], certified_anchor_frame_id=700, certified_anchor_timestamp_ms=0, cumulative_offset=1)
        reverse_step = certified_step_request(701, 33, [-1], certified_anchor_frame_id=700, certified_anchor_timestamp_ms=0, cumulative_offset=1)
        self.assertEqual((second_step.anchor_frame_id, second_step.anchor_timestamp_ms, second_step.effective_offsets), (700, 0, (2,)))
        self.assertEqual(reverse_step.effective_offsets, (0,))
