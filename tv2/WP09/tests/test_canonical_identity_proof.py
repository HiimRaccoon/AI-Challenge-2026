"""TDD coverage for E4-1A's independent canonical-identity boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wp09.contracts import CoarseCandidate, RefinementContext
from wp09.mapping import (
    CanonicalAnchor,
    CanonicalFrameRecord,
    ExactFrameResolver,
    InMemoryAnchorRegistry,
    InMemoryCanonicalFrameAuthority,
    MediaIdentity,
    ProducerCompatibility,
    RawDecodedFrame,
)


CTX = RefinementContext("run-1", "media/V1", "mapping/V1", "pyav", "n/a", "config-1")


class Decoder:
    def duration_ms(self, video_path: Path) -> int:
        return 100

    def raw_frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None):
        return (RawDecodedFrame(100, "1/1000", 0), RawDecodedFrame(133, "1/1000", 33))


def anchor() -> CanonicalAnchor:
    return CanonicalAnchor("V1", 700, 100, 0, CTX, identity_guaranteed=True)


def authority(path: Path, *, digest: str | None = None, producer: ProducerCompatibility | None = None) -> InMemoryCanonicalFrameAuthority:
    digest = digest or hashlib.sha256(path.read_bytes()).hexdigest()
    media = MediaIdentity("V1", path, digest, "1/1000", CTX)
    producer = producer or ProducerCompatibility("producer-global-decode-order-v1", "producer-global-decode-order-v1", certified=True)
    records = (
        CanonicalFrameRecord("V1", 700, 100, 0, "1/1000", CTX, media, producer),
        CanonicalFrameRecord("V1", 701, 133, 33, "1/1000", CTX, media, producer),
    )
    return InMemoryCanonicalFrameAuthority(media, records)


class CanonicalIdentityProofTests(unittest.TestCase):
    def test_single_selected_anchor_is_not_independent_neighbor_proof(self) -> None:
        result = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(),))).resolve(
            CoarseCandidate("V1", 700, 0), Path("unused.mp4"), CTX, offsets=(0, 1)
        )
        self.assertEqual(result.degraded_reason, "canonical_identity_unproven")
        self.assertTrue(all(step.frame is None for step in result.steps))

    def test_missing_bracketing_selected_anchor_is_not_mislabeled_as_proof(self) -> None:
        result = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(),))).resolve(
            CoarseCandidate("V1", 700, 0), Path("unused.mp4"), CTX, offsets=(1,)
        )
        self.assertEqual(result.steps[0].degraded_reason, "canonical_identity_unproven")

    def test_direct_per_frame_authority_not_offset_arithmetic_proves_neighbor(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "V1.mp4"
            path.write_bytes(b"original-media")
            result = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(),)), authority(path)).resolve(
                CoarseCandidate("V1", 700, 0), path, CTX, offsets=(1, 0)
            )
        self.assertEqual([step.frame.frame_id for step in result.steps if step.frame], [701, 700])
        self.assertTrue(all(step.frame.mapping_guaranteed and step.frame.submission_selectable for step in result.steps if step.frame))

    def test_wrong_source_checksum_and_producer_semantics_mismatch_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "V1.mp4"
            path.write_bytes(b"original-media")
            bad_hash = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(),)), authority(path, digest="0" * 64)).resolve(
                CoarseCandidate("V1", 700, 0), path, CTX, offsets=(1,)
            )
            mismatch = ExactFrameResolver(
                Decoder(), InMemoryAnchorRegistry((anchor(),)),
                authority(path, producer=ProducerCompatibility("frame.index", "enumerate-container-decode", certified=True)),
            ).resolve(CoarseCandidate("V1", 700, 0), path, CTX, offsets=(1,))
        self.assertEqual(bad_hash.degraded_reason, "source_checksum_mismatch")
        self.assertEqual(mismatch.degraded_reason, "producer_resolver_semantics_mismatch")

    def test_duplicate_selected_anchor_is_ambiguous(self) -> None:
        result = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(), anchor()))).resolve(
            CoarseCandidate("V1", 700, 0), Path("unused.mp4"), CTX, offsets=(0,)
        )
        self.assertEqual(result.degraded_reason, "ambiguous_anchor")

    def test_cross_anchor_mismatch_is_not_an_identity_proof(self) -> None:
        inconsistent = CanonicalAnchor("V1", 999, 133, 33, CTX, identity_guaranteed=True)
        result = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(), inconsistent))).resolve(
            CoarseCandidate("V1", 700, 0), Path("unused.mp4"), CTX, offsets=(1,)
        )
        self.assertEqual(result.degraded_reason, "canonical_identity_unproven")

    def test_wrong_original_path_and_run_id_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "V1.mp4"
            other = Path(directory) / "other.mp4"
            path.write_bytes(b"original-media")
            other.write_bytes(b"other-media")
            wrong_path = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(),)), authority(path)).resolve(
                CoarseCandidate("V1", 700, 0), other, CTX, offsets=(1,)
            )
            wrong_run = RefinementContext("other-run", "media/V1", "mapping/V1", "pyav", "n/a", "config-1")
            stale = ExactFrameResolver(Decoder(), InMemoryAnchorRegistry((anchor(),)), authority(path)).resolve(
                CoarseCandidate("V1", 700, 0), path, wrong_run, offsets=(1,)
            )
        self.assertEqual(wrong_path.degraded_reason, "media_identity_mismatch")
        self.assertEqual(stale.degraded_reason, "source_mismatch")
