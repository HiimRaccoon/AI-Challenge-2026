"""Canonical PTS-to-original-frame mapping seam for decoded video frames."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .certification import RunCertification

from .contracts import ContractError, RefinementContext, RefinementUnavailable
from .decoder import DecodeBudgetExhausted, DecodedFrame


@dataclass(frozen=True)
class RawDecodedFrame:
    pts: int | None
    time_base: str
    timestamp_ms: int
    image_rgb: object | None = None


@dataclass(frozen=True)
class ProducerCompatibility:
    """Certified statement that producer and resolver use the same identity order.

    ``run_v1_batch1`` cannot provide this statement retrospectively: its WP02
    code chose ``frame.index`` when present and otherwise used the enumerate
    counter, but its selected-frame records do not record the branch.  An
    integration must therefore hand over an explicit certification rather than
    infer compatibility from the fact that both sides use PyAV.
    """

    producer_identity_semantics: str
    resolver_identity_semantics: str
    certified: bool


@dataclass(frozen=True)
class MediaIdentity:
    """Read-only provenance for one original video, supplied by a registry."""

    video_id: str
    original_video_path: Path
    source_sha256: str
    time_base: str
    context: RefinementContext


class TrustedMediaValidator:
    """Lazy integrity validation keyed by immutable-enough source identity.

    The complete SHA-256 is computed once per unchanged file, not once per
    next/previous-frame interaction.  Any path, size, or mtime change forces
    a new validation.
    """
    def __init__(self, cache_path: Path | None = None) -> None:
        self._trusted: dict[tuple[str, int, int, str], bool] = {}
        configured = os.environ.get("WP09_TRUST_CACHE_PATH")
        self._cache_path = cache_path or (Path(configured) if configured else None)
        self.hash_count = 0

    def verify(self, path: Path, expected_sha256: str) -> bool:
        try:
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
        except OSError:
            return False
        key = (str(resolved), stat.st_size, stat.st_mtime_ns, expected_sha256.lower())
        if key in self._trusted:
            return self._trusted[key]
        disk = self._load_disk_cache()
        encoded_key = self._encode_key(key)
        if disk.get(encoded_key) is True:
            self._trusted[key] = True
            return True
        self.hash_count += 1
        try:
            value = _sha256(resolved) == expected_sha256.lower()
            after = resolved.stat()
            value = value and (after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        except OSError:
            value = False
        self._trusted[key] = value
        if value:
            disk[encoded_key] = True
            self._store_disk_cache(disk)
        return value

    def unchanged(self, path: Path, expected_sha256: str) -> bool:
        try:
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
        except OSError:
            return False
        return (str(resolved), stat.st_size, stat.st_mtime_ns, expected_sha256.lower()) in self._trusted

    @staticmethod
    def _encode_key(key: tuple[str, int, int, str]) -> str:
        return json.dumps(key, separators=(",", ":"))

    def _load_disk_cache(self) -> dict[str, bool]:
        if self._cache_path is None:
            return {}
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _store_disk_cache(self, value: dict[str, bool]) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            temporary.replace(self._cache_path)
        except OSError:
            # A cache write must never manufacture trust; the in-process value
            # remains valid only for this verified source identity.
            return


@dataclass(frozen=True)
class CanonicalFrameRecord:
    """An already-authoritative original-frame identity (never generated here)."""

    video_id: str
    frame_id: int
    pts: int
    timestamp_ms: int
    time_base: str
    context: RefinementContext
    media: MediaIdentity
    producer_compatibility: ProducerCompatibility


class CanonicalFrameAuthority(Protocol):
    """Read-only independent authority for raw-original-frame identities.

    Implementations may consume an existing authoritative per-frame mapping,
    but must not create, persist or backfill one.  The returned frame ID is the
    direct authority value; it is never derived from an anchor plus an offset.
    """

    def record_for(self, video_id: str, pts: int, time_base: str, context: RefinementContext) -> CanonicalFrameRecord | None: ...


class InMemoryCanonicalFrameAuthority:
    """Bounded/read-only adapter for externally supplied mapping records.

    This exists for adapters and tests.  It owns no file format and writes no
    mapping, so it cannot become a persistent full-frame index.
    """

    def __init__(self, media: MediaIdentity, records: Sequence[CanonicalFrameRecord]) -> None:
        self.media = media
        self._records = tuple(records)

    def record_for(self, video_id: str, pts: int, time_base: str, context: RefinementContext) -> CanonicalFrameRecord | None:
        matches = tuple(record for record in self._records if record.video_id == video_id and record.pts == pts)
        if len(matches) != 1:
            return None
        record = matches[0]
        if record.time_base != time_base or record.context != context:
            return None
        return record


@dataclass(frozen=True)
class ResolvedFrameIdentity:
    frame_id: int
    timestamp_ms: int
    pts: int


@dataclass(frozen=True)
class CanonicalAnchor:
    """A selected original frame whose global identity was supplied upstream.

    This is deliberately an anchor-only record.  It is not a generated
    full-frame mapping and it never turns a PTS, keyframe sequence or decoder
    ordinal into a submission identity by itself.
    """

    video_id: str
    frame_id: int
    pts: int | None
    timestamp_ms: int
    context: RefinementContext
    identity_guaranteed: bool

    def __post_init__(self) -> None:
        if not self.video_id or self.frame_id < 0 or self.timestamp_ms < 0:
            raise ContractError("canonical anchor fields are invalid")
        if self.pts is not None and (isinstance(self.pts, bool) or not isinstance(self.pts, int) or self.pts < 0):
            raise ContractError("canonical anchor pts is invalid")
        if not isinstance(self.context, RefinementContext):
            raise ContractError("canonical anchor context is required")


@dataclass(frozen=True)
class ExactFrame:
    """An original-frame inspection result with explicit selection authority."""

    video_id: str
    frame_id: int
    timestamp_ms: int
    pts: int
    time_base: str
    preprocess_run_id: str
    media_record_ref: str
    mapping_ref: str
    mapping_guaranteed: bool
    submission_selectable: bool
    identity_source: str = "validated_original_stream"
    degraded_reason: str | None = None
    media_identity_verified: bool = False
    producer_compatibility_verified: bool = False
    certification_id: str | None = None
    certification_report_sha256: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class ExactFrameStep:
    offset: int
    frame: ExactFrame | None
    degraded_reason: str | None = None


@dataclass(frozen=True)
class ExactFrameResolution:
    """A bounded set of requested offsets, including per-step failure detail."""

    video_id: str
    anchor_frame_id: int
    steps: tuple[ExactFrameStep, ...]
    degraded_reason: str | None = None
    provenance_mode: str = "live"

    @property
    def submission_selectable(self) -> bool:
        return bool(self.steps) and all(step.frame is not None and step.frame.submission_selectable for step in self.steps)


class FrameMappingResolver(Protocol):
    def resolve_frame(
        self, video_id: str, pts: int, time_base: str, context: RefinementContext
    ) -> ResolvedFrameIdentity: ...


class RawVideoDecoder(Protocol):
    def duration_ms(self, video_path: Path) -> int: ...

    def raw_frames_between(
        self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None
    ) -> tuple[RawDecodedFrame, ...]: ...


class CanonicalAnchorRegistry(Protocol):
    """Reads only handed-over selected-anchor mapping evidence."""

    def anchors_for(self, video_id: str, context: RefinementContext) -> tuple[CanonicalAnchor, ...]: ...


class InMemoryAnchorRegistry:
    """Small bounded registry implementation used by adapters and tests.

    Callers may populate it from TV1's existing selected-frame records.  It
    owns no persistence and therefore cannot become a backfill index.
    """

    def __init__(self, anchors: Sequence[CanonicalAnchor]) -> None:
        self._anchors = tuple(anchors)

    def anchors_for(self, video_id: str, context: RefinementContext) -> tuple[CanonicalAnchor, ...]:
        values = tuple(anchor for anchor in self._anchors if anchor.video_id == video_id)
        if not values:
            raise RefinementUnavailable("anchor_not_found")
        if any(anchor.context != context for anchor in values):
            raise RefinementUnavailable("source_mismatch")
        return values


class ExactFrameResolver:
    """Resolve bounded frames through one complete fail-closed proof chain.

    Legacy callers may still supply an independent frame authority. Production
    uses a valid run-level decoder-semantics certificate, source proof, a
    reidentified selected anchor, and actual consecutive original decode.
    Cross-anchor agreement is supporting validation only; it is never proof.
    """

    def __init__(self, raw_decoder: RawVideoDecoder, anchors: CanonicalAnchorRegistry, authority: CanonicalFrameAuthority | None = None, *, certification: RunCertification | None = None, media: MediaIdentity | None = None, media_validator: TrustedMediaValidator | None = None, max_window_ms: int = 2_000, max_decoded_frames: int = 256) -> None:
        if max_window_ms <= 0 or max_decoded_frames <= 0:
            raise ContractError("exact-frame bounds must be positive")
        self._raw_decoder = raw_decoder
        self._anchors = anchors
        self._authority = authority
        self._certification = certification
        self._media = media
        self._media_validator = media_validator or TrustedMediaValidator()
        self._max_window_ms = max_window_ms
        self._max_decoded_frames = max_decoded_frames

    def resolve(
        self,
        candidate: "CoarseCandidate",
        video_path: Path,
        context: RefinementContext,
        *,
        offsets: Sequence[int] = (0,),
    ) -> ExactFrameResolution:
        from .contracts import CoarseCandidate

        if not isinstance(candidate, CoarseCandidate) or not offsets or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets):
            raise ContractError("exact-frame request is invalid")
        try:
            anchors = self._anchors.anchors_for(candidate.video_id, context)
        except RefinementUnavailable as exc:
            return self._failed(candidate, offsets, exc.reason)
        selected = tuple(item for item in anchors if item.frame_id == candidate.frame_id)
        if len(selected) > 1:
            return self._failed(candidate, offsets, "ambiguous_anchor")
        if not selected:
            return self._failed(candidate, offsets, "anchor_not_found")
        anchor = selected[0]
        if anchor.timestamp_ms != candidate.timestamp_ms:
            return self._failed(candidate, offsets, "mapping_mismatch")
        if anchor.pts is None or not anchor.identity_guaranteed:
            return self._failed(candidate, offsets, "canonical_identity_unproven")

        try:
            duration_ms = self._raw_decoder.duration_ms(video_path)
            if duration_ms < anchor.timestamp_ms:
                return self._failed(candidate, offsets, "mapping_mismatch")
            start_ms = max(0, anchor.timestamp_ms - self._max_window_ms)
            end_ms = min(duration_ms, anchor.timestamp_ms + self._max_window_ms)
            # Infinity disables sampling; this is a bounded physical decode,
            # not a nominal-FPS estimate.
            raw_frames = self._raw_decoder.raw_frames_between(video_path, start_ms, end_ms, float("inf"), self._max_decoded_frames)
        except Exception:
            return self._failed(candidate, offsets, "decode_failure")
        anomaly = self._validate_raw_stream(raw_frames)
        if anomaly is not None:
            return self._failed(candidate, offsets, anomaly)
        anchor_indexes = [index for index, frame in enumerate(raw_frames) if frame.pts == anchor.pts]
        if len(anchor_indexes) != 1:
            return self._failed(candidate, offsets, "ambiguous_anchor" if anchor_indexes else "anchor_not_found")
        anchor_index = anchor_indexes[0]
        raw_anchor = raw_frames[anchor_index]
        if raw_anchor.timestamp_ms != anchor.timestamp_ms:
            return self._failed(candidate, offsets, "mapping_mismatch")
        truncated = len(raw_frames) >= self._max_decoded_frames
        if self._certification is not None or self._media is not None:
            return self._resolve_certified_run(candidate, anchor, raw_frames, anchor_index, video_path, context, offsets, start_ms, end_ms, duration_ms, truncated)
        if self._authority is None:
            return self._failed(candidate, offsets, "canonical_identity_unproven")
        steps: list[ExactFrameStep] = []
        for offset in offsets:
            target = anchor_index + offset
            if 0 <= target < len(raw_frames):
                raw = raw_frames[target]
                record = self._authority.record_for(candidate.video_id, raw.pts, raw.time_base, context)
                proof_error = self._validate_independent_record(record, raw, video_path, context)
                if proof_error is not None:
                    steps.append(ExactFrameStep(offset, None, proof_error))
                    continue
                assert record is not None
                steps.append(ExactFrameStep(offset, ExactFrame(
                    video_id=record.video_id,
                    frame_id=record.frame_id,
                    timestamp_ms=record.timestamp_ms,
                    pts=record.pts,
                    time_base=record.time_base,
                    preprocess_run_id=context.preprocess_run_id,
                    media_record_ref=context.media_record_ref,
                    mapping_ref=context.mapping_ref,
                    mapping_guaranteed=True,
                    submission_selectable=True,
                    identity_source="authoritative_per_frame_mapping",
                    media_identity_verified=True,
                    producer_compatibility_verified=True,
                )))
                continue
            boundary = (offset < 0 and start_ms == 0) or (offset > 0 and end_ms == duration_ms)
            reason = "boundary" if boundary and not truncated else "canonical_identity_unproven"
            steps.append(ExactFrameStep(offset, None, reason))
        failures = tuple(step.degraded_reason for step in steps if step.frame is None and step.degraded_reason)
        # Preserve a single top-level reason when every requested frame failed;
        # callers must not mistake a structurally valid empty result for proof.
        degraded_reason = failures[0] if failures and len(failures) == len(steps) else None
        return ExactFrameResolution(candidate.video_id, candidate.frame_id, tuple(steps), degraded_reason)

    def _resolve_certified_run(self, candidate: "CoarseCandidate", anchor: CanonicalAnchor, raw_frames: Sequence[RawDecodedFrame], anchor_index: int, video_path: Path, context: RefinementContext, offsets: Sequence[int], start_ms: int, end_ms: int, duration_ms: int, truncated: bool) -> ExactFrameResolution:
        cert, media = self._certification, self._media
        if cert is None or not cert.authorizes_run(context.preprocess_run_id) or not cert.runtime_compatible():
            return self._failed(candidate, offsets, "run_certification_unproven")
        if media is None or media.video_id != candidate.video_id or media.context != context or media.time_base != raw_frames[anchor_index].time_base:
            return self._failed(candidate, offsets, "media_identity_mismatch")
        try:
            if video_path.resolve(strict=True) != media.original_video_path.resolve(strict=True):
                return self._failed(candidate, offsets, "media_identity_mismatch")
        except OSError:
            return self._failed(candidate, offsets, "media_identity_unproven")
        if not self._media_validator.verify(video_path, media.source_sha256):
            return self._failed(candidate, offsets, "source_checksum_mismatch")
        # A nearby selected anchor is a consistency check only. Its absence is
        # intentionally not a rejection condition.
        for index, raw in enumerate(raw_frames):
            for other in self._anchors.anchors_for(candidate.video_id, context):
                if other is not anchor and other.pts == raw.pts and (
                    other.frame_id != anchor.frame_id + (index - anchor_index)
                    or other.timestamp_ms != raw.timestamp_ms
                ):
                    return self._failed(candidate, offsets, "second_anchor_mismatch")
        if not self._media_validator.unchanged(video_path, media.source_sha256):
            return self._failed(candidate, offsets, "source_mutated_during_decode")
        steps: list[ExactFrameStep] = []
        for offset in offsets:
            target = anchor_index + offset
            if not 0 <= target < len(raw_frames):
                boundary = (offset < 0 and start_ms == 0) or (offset > 0 and end_ms == duration_ms)
                steps.append(ExactFrameStep(offset, None, "boundary" if boundary and not truncated else "bounded_decode_incomplete"))
                continue
            raw = raw_frames[target]
            if raw.time_base != media.time_base:
                steps.append(ExactFrameStep(offset, None, "media_identity_mismatch"))
                continue
            # This arithmetic is the final consequence of: certified run
            # semantics + validated source + reidentified anchor + actual
            # consecutive original decode. It is never used as standalone ID proof.
            frame_id = anchor.frame_id + (target - anchor_index)
            if frame_id < 0:
                steps.append(ExactFrameStep(offset, None, "boundary"))
                continue
            steps.append(ExactFrameStep(offset, ExactFrame(
                video_id=candidate.video_id, frame_id=frame_id, timestamp_ms=raw.timestamp_ms,
                pts=raw.pts, time_base=raw.time_base, preprocess_run_id=context.preprocess_run_id,
                media_record_ref=context.media_record_ref, mapping_ref=context.mapping_ref,
                mapping_guaranteed=True, submission_selectable=True,
                identity_source="certified_run_consecutive_original_decode",
                media_identity_verified=True, producer_compatibility_verified=True,
                certification_id=cert.certification_id,
                certification_report_sha256=cert.certification_report_sha256,
                source_sha256=media.source_sha256,
            )))
        failures = tuple(s.degraded_reason for s in steps if s.frame is None and s.degraded_reason)
        return ExactFrameResolution(candidate.video_id, candidate.frame_id, tuple(steps), failures[0] if failures and len(failures) == len(steps) else None)

    @staticmethod
    def _validate_independent_record(record: CanonicalFrameRecord | None, raw: RawDecodedFrame, video_path: Path, context: RefinementContext) -> str | None:
        if record is None:
            return "canonical_identity_unproven"
        if record.pts != raw.pts or record.timestamp_ms != raw.timestamp_ms or record.time_base != raw.time_base:
            return "mapping_mismatch"
        if record.context != context or record.media.context != context:
            return "source_mismatch"
        compatibility = record.producer_compatibility
        if not compatibility.certified:
            return "producer_semantics_unproven"
        if compatibility.producer_identity_semantics != compatibility.resolver_identity_semantics:
            return "producer_resolver_semantics_mismatch"
        media = record.media
        try:
            actual_path = video_path.resolve(strict=True)
            expected_path = media.original_video_path.resolve(strict=True)
        except (OSError, RuntimeError):
            return "media_identity_unproven"
        if actual_path != expected_path or media.video_id != record.video_id or media.time_base != raw.time_base:
            return "media_identity_mismatch"
        try:
            digest = _sha256(actual_path)
        except OSError:
            return "media_identity_unproven"
        if digest != media.source_sha256:
            return "source_checksum_mismatch"
        return None

    @staticmethod
    def _validate_raw_stream(frames: Sequence[RawDecodedFrame]) -> str | None:
        previous_pts: int | None = None
        previous_timestamp: int | None = None
        for frame in frames:
            if frame.pts is None or isinstance(frame.pts, bool) or not isinstance(frame.pts, int):
                return "pts_unavailable"
            if not frame.time_base:
                return "unsupported_decoder_condition"
            if previous_pts is not None and frame.pts == previous_pts:
                return "duplicate_pts"
            if previous_pts is not None and frame.pts < previous_pts:
                return "non_monotonic_pts"
            if previous_timestamp is not None and frame.timestamp_ms < previous_timestamp:
                return "non_monotonic_pts"
            previous_pts, previous_timestamp = frame.pts, frame.timestamp_ms
        return None

    @staticmethod
    def _failed(candidate: "CoarseCandidate", offsets: Sequence[int], reason: str) -> ExactFrameResolution:
        return ExactFrameResolution(candidate.video_id, candidate.frame_id, tuple(ExactFrameStep(offset, None, reason) for offset in offsets), reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


class MappedVideoDecoder:
    """Adapts raw PTS frames through the sole canonical mapping resolver."""

    def __init__(self, raw_decoder: RawVideoDecoder, resolver: FrameMappingResolver, *, video_id: str, context: RefinementContext) -> None:
        self._raw_decoder = raw_decoder
        self._resolver = resolver
        self._video_id = video_id
        self._context = context

    def duration_ms(self, video_path: Path) -> int:
        return self._raw_decoder.duration_ms(video_path)

    def frames_between(self, video_path: Path, start_ms: int, end_ms: int, max_fps: float, max_frames: int | None = None) -> tuple[DecodedFrame, ...]:
        frames: list[DecodedFrame] = []
        try:
            raw_frames = self._raw_decoder.raw_frames_between(video_path, start_ms, end_ms, max_fps, max_frames)
        except DecodeBudgetExhausted:
            raise
        except Exception as exc:
            raise RefinementUnavailable("decode_failure") from exc
        for raw in raw_frames:
            try:
                identity = self._resolver.resolve_frame(self._video_id, raw.pts, raw.time_base, self._context)
            except Exception as exc:
                raise RefinementUnavailable("mapping_failure") from exc
            if identity.pts != raw.pts or identity.timestamp_ms != raw.timestamp_ms:
                raise RefinementUnavailable("mapping_failure")
            frames.append(DecodedFrame(identity.frame_id, raw.pts, raw.time_base, raw.timestamp_ms, raw.image_rgb))
        return tuple(frames)
    mapping_guaranteed = True
