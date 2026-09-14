"""State-aware verification and reranking for existing TRAKE hypotheses."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
import unicodedata
from typing import Sequence

from .contracts import SearchCandidate
from .trake_event_parser import EventState, parse_event_graph
from .visual_verifier import VisualVerification


@dataclass(frozen=True)
class TrakeVerification:
    event_scores: tuple[float, ...]
    entity_continuity: float
    state_consistency: float
    order_consistent: bool
    story_consistency: float
    modality_support: float
    micro_clip_support: float
    clip_manual_required: bool
    whole_video_prior: float
    implausible_gap_penalty: float
    contradiction_penalty: float
    final_sequence_score: float
    visual_verifier_available: bool = False
    visual_frame_ids: tuple[tuple[int, ...], ...] = ()
    visual_verifier_reasons: tuple[str | None, ...] = ()

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["event_scores"] = [round(value, 6) for value in self.event_scores]
        payload["visual_frame_ids"] = [list(value) for value in self.visual_frame_ids]
        payload["visual_verifier_reasons"] = list(self.visual_verifier_reasons)
        for key, value in tuple(payload.items()):
            if isinstance(value, float):
                payload[key] = round(value, 6)
        return payload


def _normalized_event_scores(candidates: Sequence[SearchCandidate]) -> tuple[float, ...]:
    raw = [max(0.0, float(candidate.score or 0.0)) for candidate in candidates]
    return tuple(1.0 - exp(-8.0 * value) for value in raw)


def _fold(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def _candidate_modalities(candidate: SearchCandidate) -> set[str]:
    modalities = {str(item).casefold() for item in candidate.provenance_sources}
    modalities.update(str(key).casefold() for key in candidate.model_scores)
    if candidate.source != "fusion":
        modalities.add(candidate.source.casefold())
    return modalities


def _event_modality_support(event: EventState, candidate: SearchCandidate) -> float:
    modalities = _candidate_modalities(candidate)
    folded = _fold(event.text)
    expected = {"visual"}
    if any(cue in folded for cue in ("say", "ask", "speak", "noi", "hoi")):
        expected.add("asr")
    if any(cue in folded for cue in ("text", "sign", "label", "chu", "bien")):
        expected.add("ocr")
    if event.objects:
        expected.add("object")
    if not modalities:
        return 0.25
    return len(expected & modalities) / len(expected)


def _entity_continuity(graph: Sequence[EventState]) -> float:
    references = [event for event in graph[1:] if event.refers_to_previous_entity]
    if not references:
        return 0.5
    score = 0.0
    known_entities: set[str] = set(graph[0].entities)
    for event in graph[1:]:
        if event.refers_to_previous_entity:
            # Explicit entities provide strong continuity. With only a pronoun,
            # remain neutral rather than claiming identity tracking exists.
            score += 1.0 if known_entities and set(event.entities) & known_entities else (0.6 if known_entities else 0.5)
        known_entities.update(event.entities)
    return min(0.6, score / len(references))


def _state_consistency(graph: Sequence[EventState]) -> float:
    if not graph:
        return 0.0
    score = 1.0
    holding = False
    for event in graph:
        if event.state_transition == "holding":
            holding = True
        elif event.state_transition == "released":
            if not holding:
                score -= 0.25
            holding = False
    # Text can expose a contradiction; it cannot prove a state transition
    # actually occurred in the retrieved video.
    return max(0.0, min(0.6, score))


def _micro_clip_support(
    graph: Sequence[EventState], candidates: Sequence[SearchCandidate],
    visual_verifications: Sequence[VisualVerification] | None = None,
) -> tuple[float, bool]:
    # Unknown verbs still require observation. Never interpret parser failure
    # as perfect action evidence or ignore those slots in the average.
    action_pairs = list(zip(graph, candidates))
    if not action_pairs:
        return 0.0, True

    scores: list[float] = []
    verification_by_candidate = {
        id(candidate): verification
        for candidate, verification in zip(candidates, visual_verifications or ())
    }
    for _, candidate in action_pairs:
        visual = verification_by_candidate.get(id(candidate))
        if visual is not None and visual.available and visual.score is not None:
            scores.append(max(0.0, min(1.0, visual.score)))
            continue
        explicit = candidate.provenance.get("clip_verification_score")
        if not isinstance(explicit, (int, float)) or isinstance(explicit, bool):
            explicit = candidate.provenance.get("micro_clip_score")
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            scores.append(max(0.0, min(1.0, float(explicit))))
            continue
        if (
            candidate.window_start_ms is not None
            and candidate.window_end_ms is not None
            and candidate.window_end_ms > candidate.window_start_ms
        ):
            # An upstream-owned temporal window makes clip verification
            # possible, but does not by itself prove the action semantics.
            scores.append(0.25)
        else:
            scores.append(0.2)
    support = sum(scores) / len(scores)
    return support, any(score < 0.5 for score in scores)


def verify_sequence(
    events: Sequence[str],
    candidates: Sequence[SearchCandidate],
    *,
    whole_video_rank: int = 1,
    visual_verifications: Sequence[VisualVerification] | None = None,
) -> TrakeVerification:
    """Score an existing canonical path; never creates or changes candidates."""
    graph = parse_event_graph(events)
    if len(graph) != len(candidates) or not candidates:
        raise ValueError("events and candidates must be non-empty and have equal length")

    order_consistent = all(
        left.video_id == right.video_id and left.timestamp_ms < right.timestamp_ms
        for left, right in zip(candidates, candidates[1:])
    )
    event_scores = _normalized_event_scores(candidates)
    event_support = sum(event_scores) / len(event_scores)
    entity_continuity = _entity_continuity(graph)
    state_consistency = _state_consistency(graph)
    modality_support = sum(
        _event_modality_support(event, candidate)
        for event, candidate in zip(graph, candidates)
    ) / len(candidates)
    whole_video_prior = 1.0 / max(1, whole_video_rank)
    if visual_verifications is not None and len(visual_verifications) != len(candidates):
        raise ValueError("visual verifications and candidates must have equal length")
    micro_clip_support, clip_manual_required = _micro_clip_support(
        graph, candidates, visual_verifications,
    )

    gaps = [
        right.timestamp_ms - left.timestamp_ms
        for left, right in zip(candidates, candidates[1:])
    ]
    # Sequential events separated by more than five minutes become
    # increasingly implausible, without imposing a hard duration assumption.
    gap_penalty = sum(
        1.0 - exp(-max(0, gap - 300_000) / 300_000)
        for gap in gaps
    ) / max(1, len(gaps))
    visual_contradictions = [
        item for item in visual_verifications or ()
        if item.available and item.contradiction
    ]
    visual_contradiction_penalty = len(visual_contradictions) / len(candidates)
    contradiction_penalty = max(
        0.0 if order_consistent else 1.0,
        visual_contradiction_penalty,
    )
    story_consistency = (
        0.45 * entity_continuity
        + 0.35 * state_consistency
        + 0.20 * (1.0 - gap_penalty)
    )
    final_score = (
        0.45 * event_support
        + 0.07 * whole_video_prior
        + 0.12 * entity_continuity
        + 0.09 * state_consistency
        + 0.07 * modality_support
        + 0.10 * story_consistency
        + 0.10 * micro_clip_support
        - 0.10 * gap_penalty
        - 0.50 * contradiction_penalty
    )
    return TrakeVerification(
        event_scores=event_scores,
        entity_continuity=entity_continuity,
        state_consistency=state_consistency,
        order_consistent=order_consistent,
        story_consistency=max(0.0, min(1.0, story_consistency)),
        modality_support=max(0.0, min(1.0, modality_support)),
        micro_clip_support=max(0.0, min(1.0, micro_clip_support)),
        clip_manual_required=clip_manual_required,
        whole_video_prior=max(0.0, min(1.0, whole_video_prior)),
        implausible_gap_penalty=max(0.0, min(1.0, gap_penalty)),
        contradiction_penalty=contradiction_penalty,
        final_sequence_score=max(0.0, min(1.0, final_score)),
        visual_verifier_available=any(item.available for item in visual_verifications or ()),
        visual_frame_ids=tuple(item.frame_ids for item in visual_verifications or ()),
        visual_verifier_reasons=tuple(item.reason for item in visual_verifications or ()),
    )
