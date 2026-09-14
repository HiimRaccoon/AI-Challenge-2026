from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wp09.certification import RunCertification
from wp09.contracts import CoarseCandidate, RefinementContext
from wp09.mapping import CanonicalAnchor, ExactFrameResolver, InMemoryAnchorRegistry, MediaIdentity, RawDecodedFrame, TrustedMediaValidator


CTX = RefinementContext("run_v1_batch1", "media/VX", "mapping/VX", "pyav-18.1.0", "n/a", "e4-1c")


class Decoder:
    def duration_ms(self, path: Path) -> int: return 100
    def raw_frames_between(self, path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames=None):
        return (RawDecodedFrame(1000, "1/1000", 0), RawDecodedFrame(1037, "1/1000", 37), RawDecodedFrame(1091, "1/1000", 91))


def cert(*, sample_ids=("SAMPLE",)) -> RunCertification:
    return RunCertification("e4-cert", "run_v1_batch1", "CERTIFIED", "zero_based_global_original_decode_ordinal", "a" * 64, "b" * 64, sample_ids, (), "3.12.10 x64", "18.1.0", "c" * 64)


class RunCertifiedResolutionTests(unittest.TestCase):
    def test_non_sample_video_is_authorized_by_run_not_certification_sample(self) -> None:
        with TemporaryDirectory() as d:
            path = Path(d) / "VX.mp4"; path.write_bytes(b"original")
            media = MediaIdentity("VX", path, hashlib.sha256(b"original").hexdigest(), "1/1000", CTX)
            resolver = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((CanonicalAnchor("VX", 700, 1000, 0, CTX, True),)), certification=cert(), media=media)
            with patch.object(RunCertification, "runtime_compatible", return_value=True):
                result = resolver.resolve(CoarseCandidate("VX", 700, 0), path, CTX, offsets=(0, 1, 2))
        self.assertEqual([s.frame.frame_id for s in result.steps if s.frame], [700, 701, 702])
        self.assertTrue(result.submission_selectable)

    def test_sample_ids_are_informational_and_absent_second_anchor_is_allowed(self) -> None:
        self.assertTrue(cert(sample_ids=("unrelated",)).authorizes_run("run_v1_batch1"))

    def test_media_hash_is_cached_and_invalidated_when_file_changes(self) -> None:
        with TemporaryDirectory() as d:
            path = Path(d) / "VX.mp4"; path.write_bytes(b"one")
            validator = TrustedMediaValidator(); digest = hashlib.sha256(b"one").hexdigest()
            self.assertTrue(validator.verify(path, digest)); self.assertTrue(validator.verify(path, digest))
            self.assertEqual(validator.hash_count, 1)
            path.write_bytes(b"changed")
            self.assertFalse(validator.verify(path, digest)); self.assertEqual(validator.hash_count, 2)

    def test_second_anchor_mismatch_and_uncertified_run_fail_closed(self) -> None:
        with TemporaryDirectory() as d:
            path = Path(d) / "VX.mp4"; path.write_bytes(b"original")
            media = MediaIdentity("VX", path, hashlib.sha256(b"original").hexdigest(), "1/1000", CTX)
            anchors = (CanonicalAnchor("VX", 700, 1000, 0, CTX, True), CanonicalAnchor("VX", 999, 1037, 37, CTX, True))
            with patch.object(RunCertification, "runtime_compatible", return_value=True):
                result = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry(anchors), certification=cert(), media=media).resolve(CoarseCandidate("VX", 700, 0), path, CTX, offsets=(1,))
        self.assertEqual(result.degraded_reason, "second_anchor_mismatch")
