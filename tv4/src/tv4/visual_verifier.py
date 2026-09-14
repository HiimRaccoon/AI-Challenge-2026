"""Optional frame-grounded visual verification for QA and TRAKE.

The verifier is deliberately outside retrieval/fusion.  It consumes only
canonical frame identities and derivative paths supplied by TV1, never
creates a frame ID, and fails closed to an unavailable result when the local
VLM or an image is unavailable.
"""
from __future__ import annotations

import base64
from decimal import Decimal, InvalidOperation
from io import BytesIO
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from .contracts import EvidencePack, SearchCandidate
from .media_identity import resolve_derivative_image_path
from .qa_query_plan import QaQueryPlan


@dataclass(frozen=True)
class VisualFrame:
    video_id: str
    frame_id: int
    timestamp_ms: int
    path: Path


@dataclass(frozen=True)
class VisualVerification:
    available: bool
    score: float | None
    contradiction: bool = False
    provider: str = "disabled"
    model: str | None = None
    reason: str | None = None
    frame_ids: tuple[int, ...] = ()
    rationale: str | None = None

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        if self.score is not None:
            payload["score"] = round(self.score, 6)
        payload["frame_ids"] = list(self.frame_ids)
        return payload


@dataclass(frozen=True)
class VisualQaAnswer:
    available: bool
    answer: str = ""
    score: float | None = None
    observed_values: tuple[str, ...] = ()
    provider: str = "disabled"
    model: str | None = None
    reason: str | None = None
    frame_ids: tuple[int, ...] = ()

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        if self.score is not None:
            payload["score"] = round(self.score, 6)
        payload["observed_values"] = list(self.observed_values)
        payload["frame_ids"] = list(self.frame_ids)
        return payload


Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def _environment_number(name: str, default: int | float, converter: Callable[[str], Any]) -> int | float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = converter(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if isinstance(parsed, bool) or not isinstance(parsed, (int, float)) or parsed <= 0:
        return default
    return parsed


def _default_transport(url: str, payload: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310 -- deployment-configured local VLM endpoint
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except OSError:
            detail = ""
        raise RuntimeError(f"visual verifier request failed ({exc.code}): {detail}") from exc
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("visual verifier request failed") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("visual verifier response must be an object")
    return value


def _json_object(text: str) -> Mapping[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.removeprefix("```json").removeprefix("```")
        clean = clean.removesuffix("```").strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError("visual verifier returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("visual verifier JSON must be an object")
    return payload


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("visual support score must be numeric")
    return max(0.0, min(1.0, float(value)))


def _answer_key(value: object) -> str:
    """Normalize a concise answer for independent-read agreement checks."""
    if not isinstance(value, str):
        return ""
    clean = " ".join(value.casefold().split())
    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", clean)
    if len(numbers) == 1:
        try:
            return f"number:{Decimal(numbers[0].replace(',', '.')).normalize()}"
        except InvalidOperation:
            pass
    return re.sub(r"[^a-z0-9]+", "", clean)


class OllamaVisualVerifier:
    """Bounded Ollama-compatible VLM adapter, disabled unless opted in."""

    def __init__(
        self,
        *,
        enabled: bool,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 45.0,
        max_images: int = 5,
        max_image_bytes: int = 4 * 1024 * 1024,
        max_qa_candidates: int = 3,
        max_sequence_candidates: int = 3,
        transport: Transport | None = None,
    ) -> None:
        if (
            timeout_seconds <= 0 or max_images < 1 or max_image_bytes < 1
            or max_qa_candidates < 1 or max_sequence_candidates < 1
        ):
            raise ValueError("visual verifier limits must be positive")
        self.enabled = enabled and bool(model.strip())
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_images = max_images
        self.max_image_bytes = max_image_bytes
        self.max_qa_candidates = max_qa_candidates
        self.max_sequence_candidates = max_sequence_candidates
        self._transport = transport or _default_transport

    @classmethod
    def from_environment(cls) -> "OllamaVisualVerifier":
        return cls(
            enabled=os.environ.get("TV4_VISUAL_VERIFIER_ENABLED", "0") == "1",
            model=os.environ.get("TV4_VISUAL_VERIFIER_MODEL", ""),
            base_url=os.environ.get("TV4_VISUAL_VERIFIER_BASE_URL", "http://127.0.0.1:11434"),
            timeout_seconds=float(_environment_number("TV4_VISUAL_VERIFIER_TIMEOUT_SECONDS", 45.0, float)),
            max_images=int(_environment_number("TV4_VISUAL_VERIFIER_MAX_IMAGES", 5, int)),
            max_image_bytes=int(_environment_number("TV4_VISUAL_VERIFIER_MAX_IMAGE_BYTES", 4 * 1024 * 1024, int)),
            max_qa_candidates=int(_environment_number("TV4_VISUAL_VERIFIER_MAX_QA_CANDIDATES", 3, int)),
            max_sequence_candidates=int(_environment_number("TV4_VISUAL_VERIFIER_MAX_SEQUENCES", 3, int)),
        )

    def unavailable(self, reason: str, frames: Sequence[VisualFrame] = ()) -> VisualVerification:
        return VisualVerification(
            available=False,
            score=None,
            provider="ollama-vlm" if self.enabled else "disabled",
            model=self.model or None,
            reason=reason,
            frame_ids=tuple(frame.frame_id for frame in frames),
        )

    def unavailable_answer(self, reason: str, frames: Sequence[VisualFrame] = ()) -> VisualQaAnswer:
        return VisualQaAnswer(
            available=False,
            provider="ollama-vlm" if self.enabled else "disabled",
            model=self.model or None,
            reason=reason,
            frame_ids=tuple(frame.frame_id for frame in frames),
        )

    def _chat(
        self,
        prompt: str,
        frames: Sequence[VisualFrame],
        *,
        detail_tiles: bool = False,
        label_crop: bool = False,
    ) -> Mapping[str, Any]:
        selected = tuple(frames[:1] if detail_tiles else frames[: self.max_images])
        if not self.enabled:
            raise RuntimeError("visual verifier disabled")
        if not selected:
            raise RuntimeError("canonical visual evidence unavailable")
        images: list[str] = []
        for frame in selected:
            size = frame.path.stat().st_size
            if size <= 0 or size > self.max_image_bytes:
                raise RuntimeError("visual evidence image exceeds configured bounds")
            raw = frame.path.read_bytes()
            if label_crop:
                try:
                    with Image.open(BytesIO(raw)) as source:
                        source = source.convert("RGB")
                        width, height = source.size
                        crop = source.crop((0, round(height * 0.2), width, round(height * 0.92)))
                        if crop.width < 1600:
                            crop = crop.resize(
                                (1600, round(crop.height * 1600 / crop.width)),
                                Image.Resampling.LANCZOS,
                            )
                        output = BytesIO()
                        crop.save(output, format="JPEG", quality=92, optimize=True)
                        raw = output.getvalue()
                except (OSError, UnidentifiedImageError, ValueError):
                    pass
            images.append(base64.b64encode(raw).decode("ascii"))
            if detail_tiles:
                # A chart may occupy only a small region of a canonical video
                # frame.  Send deterministic overlapping zoom tiles derived
                # from that same frame, without inventing a new frame ID.
                try:
                    with Image.open(BytesIO(raw)) as source:
                        source = source.convert("RGB")
                        width, height = source.size
                        boxes = (
                            (0, 0, round(width * 0.6), round(height * 0.6)),
                            (round(width * 0.4), 0, width, round(height * 0.6)),
                            (0, round(height * 0.4), round(width * 0.6), height),
                            (round(width * 0.4), round(height * 0.4), width, height),
                        )
                        for box in boxes[: max(0, self.max_images - 1)]:
                            tile = source.crop(box)
                            tile.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                            output = BytesIO()
                            tile.save(output, format="JPEG", quality=92, optimize=True)
                            images.append(base64.b64encode(output.getvalue()).decode("ascii"))
                except (OSError, UnidentifiedImageError, ValueError):
                    # Tests and degraded deployments may use a non-decodable
                    # placeholder.  The canonical full image remains usable.
                    pass
        response = self._transport(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "stream": False,
                # Qwen3-VL supports disabling its hidden reasoning pass.  The
                # verifier needs a small JSON decision, so a long thinking
                # trace only adds latency and memory pressure.
                "think": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 12288},
                "messages": [{"role": "user", "content": prompt, "images": images}],
            },
            self.timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        # Some Ollama/Qwen3-VL versions place structured output in the
        # dedicated ``thinking`` field even when thinking is disabled.  It is
        # still model output governed by the same JSON schema, so accept it as
        # a compatibility fallback when normal content is empty.
        if (not isinstance(content, str) or not content.strip()) and isinstance(message, Mapping):
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                content = thinking
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("visual verifier returned no content")
        return _json_object(content)

    def verify_scene(self, description: str, frames: Sequence[VisualFrame]) -> VisualVerification:
        """Check scene identity independently, without exposing the proposed answer."""
        selected = tuple(frames[:self.max_images])
        if not self.enabled or not selected:
            return self.unavailable('disabled' if not self.enabled else 'canonical_frames_unavailable', selected)
        clauses = [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+', description) if part.strip()]
        criteria = clauses[:5] + ([' '.join(clauses[5:])] if len(clauses) > 5 else [])
        prompt = (
            'Compare the ordered images with the scene description. Verify the distinctive '
            'subject, physical attributes, objects and actions, including their order when visible. '
            'The same TV programme, presenter, kitchen, watermark or broad topic is NOT enough. '
            'A shared number or caption is NOT evidence of matching scenes. '
            'Use score >= 0.7 only when distinctive described details are directly visible. '
            'Use a low score if only the broad topic matches or crucial details are not visible; '
            'set contradiction=true only for a visible conflict. Do not infer missing actions. '
            'For EVERY numbered criterion, mark matched=true only if ALL its distinctive details '
            'are visible. Unobserved details mean matched=false, even if plausible. Cite only supplied '
            'frame IDs. Return JSON only: {"score":0..1,"contradiction":true|false,'
            '"checks":[{"criterion":0,"matched":true|false,"frame_ids":[123]}],"rationale":"short"}.\n'
            f'Criteria: {json.dumps(dict(enumerate(criteria)), ensure_ascii=False)}\n'
            f'Frame IDs: {[f.frame_id for f in selected]}'
        )
        try:
            payload = self._chat(prompt, selected)
            score = _score(payload.get('score'))
            contradiction = payload.get('contradiction')
            if not isinstance(contradiction, bool):
                raise ValueError('scene contradiction must be boolean')
            checks = payload.get('checks')
            if not isinstance(checks, list) or len(checks) != len(criteria):
                raise ValueError('scene checks must cover every criterion')
            indexes, matched = set(), 0
            allowed_frames = {f.frame_id for f in selected}
            for check in checks:
                if not isinstance(check, dict):
                    raise ValueError('invalid scene check')
                index, agrees, frame_ids = check.get('criterion'), check.get('matched'), check.get('frame_ids')
                if type(index) is not int or index not in range(len(criteria)) or index in indexes or type(agrees) is not bool:
                    raise ValueError('invalid scene criterion')
                if not isinstance(frame_ids, list) or any(type(f) is not int or f not in allowed_frames for f in frame_ids):
                    raise ValueError('unrecognized scene evidence frame')
                indexes.add(index)
                matched += int(agrees and bool(frame_ids))
            # A high generic score cannot override a missing distinctive clause.
            if matched != len(criteria):
                score = min(score, 0.49 * matched / max(1, len(criteria)))
            return VisualVerification(available=True, score=score, contradiction=contradiction,
                                      provider='ollama-vlm', model=self.model,
                                      frame_ids=tuple(f.frame_id for f in selected),
                                      rationale=f'{matched}/{len(criteria)} criteria observed. ' + str(payload.get('rationale', ''))[:450])
        except Exception as exc:
            return self.unavailable(type(exc).__name__, selected)

    def verify_qa(
        self, question: str, answer: str, frames: Sequence[VisualFrame],
    ) -> VisualVerification:
        selected = tuple(frames[: self.max_images])
        if not self.enabled:
            return self.unavailable("disabled", selected)
        if not selected:
            return self.unavailable("canonical_frames_unavailable")
        prompt = (
            "You are a conservative visual fact checker. Judge whether the ordered images "
            "directly support the proposed answer to the question. Do not use outside knowledge. "
            "Return JSON only: {\"score\":0..1,\"contradiction\":true|false,\"rationale\":\"short\"}.\n"
            f"Question: {question}\nProposed answer: {answer}\n"
            f"Canonical frame IDs in image order: {[frame.frame_id for frame in selected]}"
        )
        try:
            payload = self._chat(prompt, selected)
            score = _score(payload.get("score"))
            contradiction = payload.get("contradiction")
            if not isinstance(contradiction, bool):
                raise ValueError("visual contradiction must be boolean")
            rationale = payload.get("rationale")
            return VisualVerification(
                available=True,
                score=score,
                contradiction=contradiction,
                provider="ollama-vlm",
                model=self.model,
                frame_ids=tuple(frame.frame_id for frame in selected),
                rationale=str(rationale)[:500] if rationale is not None else None,
            )
        except Exception as exc:
            return self.unavailable(type(exc).__name__, selected)

    def answer_qa(
        self,
        question: str,
        plan: QaQueryPlan,
        frames: Sequence[VisualFrame],
    ) -> VisualQaAnswer:
        """Answer only explicitly routed, bounded visual QA operations."""
        selected_limit = (
            1
            if plan.operation == "read_chart_relation"
            else self.max_images * self.max_sequence_candidates
        )
        selected = tuple(frames[:selected_limit])
        if not self.enabled:
            return self.unavailable_answer("disabled", selected)
        if not plan.requires_visual_read:
            return self.unavailable_answer("question_not_visual_readable", selected)
        if not selected:
            return self.unavailable_answer("canonical_frames_unavailable")
        if plan.operation == "set_difference":
            observed_union: set[str] = set()
            batch_scores: list[float] = []
            try:
                for batch_index in range(0, len(selected), self.max_images):
                    batch = selected[batch_index: batch_index + self.max_images]
                    prompt = (
                        "Inspect only this chronological batch sampled from the requested video interval. "
                        "List every candidate label physically attached to the described objects. Ignore "
                        "broadcast clocks, channel logos, scores, and unrelated overlay text. Return JSON "
                        "only: {\"answer\":\"\",\"observed_values\":[\"...\"],\"score\":0..1}.\n"
                        f"Question: {question}\nCandidate values: {list(plan.candidate_values)}\n"
                        f"Interval ms: {plan.interval_start_ms}..{plan.interval_end_ms}\n"
                        f"Canonical frame IDs in this batch: {[frame.frame_id for frame in batch]}"
                    )
                    payload = self._chat(prompt, batch, label_crop=True)
                    observed = payload.get("observed_values", ())
                    if not isinstance(observed, list) or not all(
                        isinstance(value, (str, int)) and not isinstance(value, bool)
                        for value in observed
                    ):
                        raise ValueError("observed_values must be a list of scalar values")
                    normalized = {
                        str(value).strip() for value in observed if str(value).strip()
                    }
                    allowed = set(plan.candidate_values)
                    if any(value not in allowed for value in normalized):
                        raise ValueError("observed value is outside the requested range")
                    observed_union.update(normalized)
                    batch_scores.append(_score(payload.get("score")))
                missing = [
                    value for value in plan.candidate_values
                    if value not in observed_union
                ]
                if len(missing) != 1:
                    return self.unavailable_answer("set_difference_not_proven", selected)
                return VisualQaAnswer(
                    available=True,
                    answer=missing[0],
                    score=sum(batch_scores) / len(batch_scores) if batch_scores else 0.0,
                    observed_values=tuple(
                        value for value in plan.candidate_values if value in observed_union
                    ),
                    provider="ollama-vlm",
                    model=self.model,
                    frame_ids=tuple(frame.frame_id for frame in selected),
                )
            except Exception as exc:
                return self.unavailable_answer(type(exc).__name__, selected)
        elif plan.operation == "read_chart_relation":
            prompt = (
                "The first image is one canonical frame; the following images are overlapping "
                "magnified tiles from that same frame. Read the requested relation between the "
                "named chart curve or category and its axes. Do not choose a nearby unrelated "
                "number, watermark, phone number, question number, or answer option. Return JSON "
                "only: {\"answer\":\"concise value or empty\",\"observed_values\":[],"
                "\"score\":0..1}. Use an empty answer when labels or the relation are not legible.\n"
                f"Question: {question}\n"
                f"Canonical frame IDs in image order: {[frame.frame_id for frame in selected]}"
            )
        elif plan.operation in {"read_visible_text", "count_visible"}:
            selected = selected[:self.max_images]
            prompt = (
                "Answer the question using only these canonical frames from one video window. "
                "Read only text on the requested object, or count the requested objects/distinct colors. "
                "Do not count repeated observations across frames as new objects. "
                "Ignore unrelated channel logos, clocks and overlays unless explicitly requested. "
                "If the target is not visible or the count/text is ambiguous, return an empty answer. "
                'Return JSON only: {"answer":"concise answer or empty","observed_values":[],"score":0..1}.\n'
                f"Question: {question}\nFrame IDs: {[frame.frame_id for frame in selected]}"
            )
        else:
            return self.unavailable_answer("unsupported_visual_operation", selected)
        try:
            payload = self._chat(
                prompt,
                selected,
                detail_tiles=plan.operation == "read_chart_relation",
            )
            if plan.operation == "read_chart_relation":
                # A single tiled read can confidently latch onto a nearby tick
                # or legend value. Require an independent full-frame read to
                # reproduce the answer before exposing it as a proposal.
                independent = self._chat(
                    prompt + "\nThis is an independent full-frame reading; do not rely on a prior guess.",
                    selected,
                    detail_tiles=False,
                )
                first_key = _answer_key(payload.get("answer"))
                second_key = _answer_key(independent.get("answer"))
                if not first_key or first_key != second_key:
                    return self.unavailable_answer(
                        "chart_reading_not_reproducible", selected,
                    )
                payload = {
                    **payload,
                    "score": min(
                        _score(payload.get("score")),
                        _score(independent.get("score")),
                    ),
                }
            answer = payload.get("answer")
            score = _score(payload.get("score"))
            observed = payload.get("observed_values", ())
            if not isinstance(answer, str) or len(answer.strip()) > 100:
                raise ValueError("visual answer must be a bounded string")
            if not isinstance(observed, list) or not all(isinstance(value, (str, int)) for value in observed):
                raise ValueError("observed_values must be a list of scalar values")
            normalized_observed = tuple(dict.fromkeys(
                str(value).strip() for value in observed if str(value).strip()
            ))
            clean_answer = answer.strip()
            if not clean_answer:
                return self.unavailable_answer("insufficient_visual_evidence", selected)
            return VisualQaAnswer(
                available=True,
                answer=clean_answer,
                score=score,
                observed_values=normalized_observed,
                provider="ollama-vlm",
                model=self.model,
                frame_ids=tuple(frame.frame_id for frame in selected),
            )
        except Exception as exc:
            return self.unavailable_answer(type(exc).__name__, selected)

    def verify_answer_source_relation(
        self,
        description: str,
        question: str,
        device: str,
        frames: Sequence[VisualFrame],
    ) -> VisualVerification:
        """Check the device/display relation before accepting numeric OCR.

        A visible number on an arbitrary screen is not enough.  The ordered
        frames must show the requested device and the display must belong to
        that device in the described event.
        """
        selected = tuple(frames[: self.max_images])
        if not self.enabled:
            return self.unavailable("disabled", selected)
        if not selected:
            return self.unavailable("canonical_frames_unavailable")
        prompt = (
            "You are a conservative visual relation verifier. Judge only these ordered images. "
            f"The required device is '{device}'. A generic monitor, machine screen, sign, label, "
            "or price board does NOT count. The displayed value must be physically attached to or "
            "clearly belong to the required device, and the surrounding event must agree with the "
            "description. Return JSON only: "
            "{\"score\":0..1,\"contradiction\":true|false,\"rationale\":\"short\"}.\n"
            f"Description: {description}\nQuestion: {question}\n"
            f"Canonical frame IDs in image order: {[frame.frame_id for frame in selected]}"
        )
        try:
            payload = self._chat(prompt, selected)
            score = _score(payload.get("score"))
            contradiction = payload.get("contradiction")
            if not isinstance(contradiction, bool):
                raise ValueError("visual contradiction must be boolean")
            rationale = payload.get("rationale")
            return VisualVerification(
                available=True,
                score=score,
                contradiction=contradiction,
                provider="ollama-vlm",
                model=self.model,
                frame_ids=tuple(frame.frame_id for frame in selected),
                rationale=str(rationale)[:500] if rationale is not None else None,
            )
        except Exception as exc:
            return self.unavailable(type(exc).__name__, selected)

    def verify_events(
        self,
        events: Sequence[str],
        frame_groups: Sequence[Sequence[VisualFrame]],
    ) -> tuple[VisualVerification, ...]:
        if len(events) != len(frame_groups):
            raise ValueError("events and visual frame groups must have equal length")
        return tuple(
            self._verify_event(event, frames)
            for event, frames in zip(events, frame_groups)
        )

    def _verify_event(self, event: str, frames: Sequence[VisualFrame]) -> VisualVerification:
        selected = tuple(frames[: self.max_images])
        if not self.enabled:
            return self.unavailable("disabled", selected)
        if not selected:
            return self.unavailable("canonical_frames_unavailable")
        prompt = (
            "You are a conservative temporal-event verifier. The images are ordered canonical "
            "keyframes from one short window. Judge whether they directly support the event, "
            "including the state change when one is claimed. Return JSON only: "
            "{\"score\":0..1,\"contradiction\":true|false,\"rationale\":\"short\"}.\n"
            f"Event: {event}\nCanonical frame IDs in image order: {[frame.frame_id for frame in selected]}"
        )
        try:
            payload = self._chat(prompt, selected)
            score = _score(payload.get("score"))
            contradiction = payload.get("contradiction")
            if not isinstance(contradiction, bool):
                raise ValueError("visual contradiction must be boolean")
            rationale = payload.get("rationale")
            return VisualVerification(
                available=True,
                score=score,
                contradiction=contradiction,
                provider="ollama-vlm",
                model=self.model,
                frame_ids=tuple(frame.frame_id for frame in selected),
                rationale=str(rationale)[:500] if rationale is not None else None,
            )
        except Exception as exc:
            return self.unavailable(type(exc).__name__, selected)


def _bounded_frames(frames: Sequence[VisualFrame], limit: int) -> tuple[VisualFrame, ...]:
    if len(frames) <= limit:
        return tuple(frames)
    if limit == 1:
        return (frames[len(frames) // 2],)
    indexes = [round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)]
    return tuple(frames[index] for index in dict.fromkeys(indexes))


def canonical_qa_frames(
    evidence: EvidencePack,
    asset_root: Path | None,
    allowed_extensions: Sequence[str],
    *,
    preprocess_run_id: str,
    limit: int = 5,
) -> tuple[VisualFrame, ...]:
    """Resolve QA images only from TV1-owned selected-frame records."""
    if asset_root is None or limit < 1:
        return ()
    resolved: list[VisualFrame] = []
    seen: set[int] = set()
    ordered = sorted(
        evidence.selected_frames,
        key=lambda frame: (
            frame.frame_id != evidence.frame_id,
            abs(frame.timestamp_ms - evidence.timestamp_ms),
            frame.timestamp_ms,
            frame.frame_id,
        ),
    )
    for frame in ordered:
        if frame.video_id != evidence.video_id or frame.frame_id in seen or not frame.keyframe_path:
            continue
        if frame.preprocess_run_id not in (None, preprocess_run_id):
            continue
        try:
            path = resolve_derivative_image_path(asset_root, frame.keyframe_path, allowed_extensions)
        except ValueError:
            continue
        seen.add(frame.frame_id)
        resolved.append(VisualFrame(frame.video_id, frame.frame_id, frame.timestamp_ms, path))
    resolved.sort(key=lambda frame: (frame.timestamp_ms, frame.frame_id))
    return _bounded_frames(resolved, limit)


def canonical_interval_frames(
    video_id: str,
    tv1: Any,
    asset_root: Path | None,
    allowed_extensions: Sequence[str],
    *,
    preprocess_run_id: str,
    start_ms: int,
    end_ms: int,
    limit: int = 8,
) -> tuple[VisualFrame, ...]:
    """Resolve evenly sampled TV1-owned frames from an explicit interval."""
    if asset_root is None or limit < 1 or start_ms < 0 or end_ms < start_ms:
        return ()
    try:
        records = tv1.frames(video_id)
    except Exception:
        return ()
    frames: list[VisualFrame] = []
    seen: set[int] = set()
    for record in records if isinstance(records, list) else ():
        if not isinstance(record, Mapping):
            continue
        frame_id = record.get("frame_id")
        timestamp_ms = record.get("timestamp_ms")
        path_value = record.get("keyframe_path")
        if (
            record.get("video_id") not in (None, video_id)
            or record.get("preprocess_run_id") not in (None, preprocess_run_id)
            or isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0
            or isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int)
            or not start_ms <= timestamp_ms <= end_ms
            or not isinstance(path_value, str) or not path_value
            or frame_id in seen
        ):
            continue
        try:
            path = resolve_derivative_image_path(asset_root, path_value, allowed_extensions)
        except ValueError:
            continue
        seen.add(frame_id)
        frames.append(VisualFrame(video_id, frame_id, timestamp_ms, path))
    frames.sort(key=lambda frame: (frame.timestamp_ms, frame.frame_id))
    return _bounded_frames(frames, limit)


def canonical_trake_frame_groups(
    candidates: Sequence[SearchCandidate],
    tv1: Any,
    asset_root: Path | None,
    allowed_extensions: Sequence[str],
    *,
    preprocess_run_id: str,
    limit_per_event: int = 5,
    context_ms: int = 2_000,
) -> tuple[tuple[VisualFrame, ...], ...]:
    """Collect ordered TV1 keyframes around candidates without deriving IDs."""
    if asset_root is None or limit_per_event < 1 or context_ms < 0:
        return tuple(() for _ in candidates)
    by_video: dict[str, Sequence[Mapping[str, Any]]] = {}
    groups: list[tuple[VisualFrame, ...]] = []
    for candidate in candidates:
        if candidate.video_id not in by_video:
            try:
                payload = tv1.frames(candidate.video_id)
                by_video[candidate.video_id] = tuple(item for item in payload if isinstance(item, Mapping))
            except Exception:
                by_video[candidate.video_id] = ()
        start_ms = candidate.window_start_ms if candidate.window_start_ms is not None else candidate.timestamp_ms - context_ms
        end_ms = candidate.window_end_ms if candidate.window_end_ms is not None else candidate.timestamp_ms + context_ms
        selected: list[VisualFrame] = []
        seen: set[int] = set()
        for record in by_video[candidate.video_id]:
            frame_id = record.get("frame_id")
            timestamp_ms = record.get("timestamp_ms")
            path_value = record.get("keyframe_path")
            if (
                record.get("video_id") not in (None, candidate.video_id)
                or record.get("preprocess_run_id") not in (None, preprocess_run_id)
                or isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0
                or isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0
                or not isinstance(path_value, str) or not path_value
                or not (start_ms <= timestamp_ms <= end_ms)
                or frame_id in seen
            ):
                continue
            try:
                path = resolve_derivative_image_path(asset_root, path_value, allowed_extensions)
            except ValueError:
                continue
            seen.add(frame_id)
            selected.append(VisualFrame(candidate.video_id, frame_id, timestamp_ms, path))
        selected.sort(key=lambda frame: (frame.timestamp_ms, frame.frame_id))
        groups.append(_bounded_frames(selected, limit_per_event))
    return tuple(groups)
