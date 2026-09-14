"""QA-only answerability ranking over already retrieved evidence.

This module deliberately has no dependency on the KIS orchestration or fusion
code.  It consumes immutable ``SearchCandidate``/``EvidencePack`` values and
returns QA-specific copies, keeping the existing KIS ranking untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
import unicodedata
from typing import Sequence

from .contracts import EvidencePack, SearchCandidate
from .qa_query_plan import build_qa_query_plan, quantity_subject


def _fold(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


_SPEECH_CUES = re.compile(
    r"\b(noi|tra loi|phat bieu|ten la gi|ai|o dau|khi nao|"
    r"say|said|tell|ask|name|who|where|when)\b"
)
_OCR_CUES = re.compile(
    r"\b(chu|bien|bang|nhan|logo|phu de|so hien thi|man hinh|mat so|"
    r"text|sign|label|caption|display|readout)\b"
)
_ACTION_CUES = re.compile(
    r"\b(dang lam gi|dua|nhan|mo|dong|cam|lay|dat|dung day|roi di|"
    r"doing|give|take|open|close|holding|leave|walk|run)\b"
)


def preferred_modalities(question: str) -> tuple[str, ...]:
    """Infer only broad evidence needs; visual remains the safe default."""
    folded = _fold(question)
    preferred: list[str] = []
    if quantity_subject(question):
        preferred.extend(('ocr', 'asr'))
    plan = build_qa_query_plan("", question)
    if plan.answer_source == "device_display":
        preferred.extend(("ocr", "visual", "object"))
    elif _SPEECH_CUES.search(folded):
        preferred.append("asr")
    if _OCR_CUES.search(folded):
        preferred.append("ocr")
    if _ACTION_CUES.search(folded):
        preferred.extend(("visual", "object"))
    if not preferred:
        preferred.extend(("visual", "object"))
    return tuple(dict.fromkeys(preferred))


@dataclass(frozen=True)
class AnswerabilityScore:
    retrieval_relevance: float
    modality_match: float
    transcript_answer_cue: float
    visual_object_answer_cue: float
    temporal_completeness: float
    semantic_answer_cue: float
    total: float

    def to_json(self) -> dict[str, float]:
        return {name: round(float(value), 6) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class RankedEvidence:
    candidate: SearchCandidate
    evidence: EvidencePack
    answerability: AnswerabilityScore


def qa_evidence_window_ms(retrieval_position: int) -> int:
    """Use broad evidence only for the strongest QA retrieval candidates.

    The policy is deliberately independent of question wording and answer
    vocabulary. It gives unseen question types nearby OCR/ASR context without
    paying the wide-window cost for the complete 30-item answerability pool.
    """
    if retrieval_position < 0:
        raise ValueError("retrieval_position must be non-negative")
    return 30_000 if retrieval_position < 16 else 3_000


def select_evidence_candidates(candidates: Sequence[SearchCandidate], limit: int = 30) -> list[SearchCandidate]:
    """Inspect distinct videos first, retaining room for other time windows."""
    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[SearchCandidate] = []
    videos: set[str] = set()
    for candidate in candidates:
        if candidate.video_id in videos:
            continue
        videos.add(candidate.video_id)
        selected.append(candidate)
        if len(selected) >= max(1, limit * 2 // 3):
            break
    identities = {(c.video_id, c.frame_id) for c in selected}
    for candidate in candidates:
        if len(selected) >= limit:
            break
        identity = (candidate.video_id, candidate.frame_id)
        if identity not in identities:
            identities.add(identity)
            selected.append(candidate)
    return selected


def _retrieval_relevance(candidate: SearchCandidate, maximum_score: float) -> float:
    score = candidate.score if candidate.score is not None else candidate.confidence
    if score is not None and maximum_score > 0:
        return max(0.0, min(1.0, float(score) / maximum_score))
    return 1.0 / max(1, candidate.rank)


_SEMANTIC_STOPWORDS = {
    "anh", "canh", "cho", "co", "cua", "da", "day", "de", "den", "duoc",
    "gan", "gi", "hoi", "ke", "la", "lai", "loai", "mot", "nay", "nhung",
    "quanh", "sau", "the", "thi", "trong", "tren", "truoc", "video", "voi",
    "what", "which", "this", "that", "the", "and", "from", "into", "near",
}
_ANIMAL_QUESTION = re.compile(
    r"\b(?:con gi|thit (?:cua )?con gi|loai (?:ca|thit|hai san) gi|"
    r"what animal|which animal|what kind of (?:fish|meat|seafood))\b"
)
_ANIMAL_ANSWER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnghêu\s+sữa\b", re.IGNORECASE), "Nghêu sữa"),
    (re.compile(r"\bnghêu\b", re.IGNORECASE), "Nghêu"),
    (re.compile(r"\bngao\b", re.IGNORECASE), "Ngao"),
    (re.compile(r"\bsò\b", re.IGNORECASE), "Sò"),
    (re.compile(r"\bhến\b", re.IGNORECASE), "Hến"),
    (re.compile(r"\bcá\s+hồi\b", re.IGNORECASE), "Cá hồi"),
    (re.compile(r"\bcá\s+ngừ\b", re.IGNORECASE), "Cá ngừ"),
    (re.compile(r"\bcá\s+thu\b", re.IGNORECASE), "Cá thu"),
    (re.compile(r"\bcá\s+chép\b", re.IGNORECASE), "Cá chép"),
    (re.compile(r"\bcá\s+lóc\b", re.IGNORECASE), "Cá lóc"),
    (re.compile(r"\bcá\s+basa\b", re.IGNORECASE), "Cá basa"),
    (re.compile(r"\bcá\s+diêu\s+hồng\b", re.IGNORECASE), "Cá diêu hồng"),
    (re.compile(r"\btôm\b", re.IGNORECASE), "Tôm"),
    (re.compile(r"\bmực\b", re.IGNORECASE), "Mực"),
    (re.compile(r"\bgà\b", re.IGNORECASE), "Gà"),
    (re.compile(r"\bbò\b", re.IGNORECASE), "Bò"),
    (re.compile(r"\bheo\b", re.IGNORECASE), "Heo"),
    (re.compile(r"\blợn\b", re.IGNORECASE), "Lợn"),
    (re.compile(r"\bvịt\b", re.IGNORECASE), "Vịt"),
    (re.compile(r"\bdê\b", re.IGNORECASE), "Dê"),
    (re.compile(r"\bcừu\b", re.IGNORECASE), "Cừu"),
    (re.compile(r"\bcua\b", re.IGNORECASE), "Cua"),
    (re.compile(r"\bốc\b", re.IGNORECASE), "Ốc"),
    (re.compile(r"\bếch\b", re.IGNORECASE), "Ếch"),
    (re.compile(r"\blươn\b", re.IGNORECASE), "Lươn"),
    (re.compile(r"\bsalmon\b", re.IGNORECASE), "Cá hồi"),
    (re.compile(r"\btuna\b", re.IGNORECASE), "Cá ngừ"),
    (re.compile(r"\b(?:shrimp|prawn)\b", re.IGNORECASE), "Tôm"),
    (re.compile(r"\bsquid\b", re.IGNORECASE), "Mực"),
    (re.compile(r"\bchicken\b", re.IGNORECASE), "Gà"),
    (re.compile(r"\b(?:beef|cow)\b", re.IGNORECASE), "Bò"),
    (re.compile(r"\b(?:pork|pig)\b", re.IGNORECASE), "Heo"),
    (re.compile(r"\bduck\b", re.IGNORECASE), "Vịt"),
    (re.compile(r"\bgoat\b", re.IGNORECASE), "Dê"),
    (re.compile(r"\b(?:lamb|sheep)\b", re.IGNORECASE), "Cừu"),
    (re.compile(r"\bcrab\b", re.IGNORECASE), "Cua"),
    (re.compile(r"\bclam(?:s)?\b", re.IGNORECASE), "Nghêu"),
    (re.compile(r"\b(?:snail|eel|frog)\b", re.IGNORECASE), "Động vật"),
    # Generic fish comes last so a specific species wins when both match.
    (re.compile(r"\bcá\b", re.IGNORECASE), "Cá"),
)


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _fold(text))
        if len(token) >= 3 and token not in _SEMANTIC_STOPWORDS
    }


def _evidence_text(evidence: EvidencePack) -> str:
    values: list[str] = [
        *evidence.asr_texts,
        *evidence.ocr_texts,
        *evidence.object_labels,
        *(item.text for item in evidence.asr_evidence),
        *(
            str(row.get("text", ""))
            for item in evidence.asr_evidence
            for row in item.context
            if isinstance(row, dict)
        ),
        *(item.raw_text for item in evidence.ocr_evidence),
        *(item.normalized_text for item in evidence.ocr_evidence),
        *(item.label for item in evidence.object_evidence),
    ]
    return " ".join(value for value in values if isinstance(value, str) and value.strip())


def extract_typed_text_answer(question: str, evidence_text: str) -> str:
    """Extract a narrow answer only when its requested type is explicit.

    Accent-sensitive matching is intentional. Folding Vietnamese would turn
    common words such as ``của`` and ``để`` into animal names ``cua`` and
    ``dê``, creating confident false positives.
    """
    if not _ANIMAL_QUESTION.search(_fold(question)):
        return ""
    normalized = unicodedata.normalize("NFC", evidence_text)
    for pattern, answer in _ANIMAL_ANSWER_PATTERNS:
        if pattern.search(normalized):
            return answer
    return ""


def extract_quantity_ocr_answer(question: str, texts: Sequence[str]) -> str:
    """Read a named entity's quantity, requiring agreement across OCR lines."""
    subject = quantity_subject(question)
    if not subject:
        return ''
    target = r'\s*'.join(re.escape(word) for word in subject.split())
    pattern = re.compile(rf'(?<![a-z]){target}\s*[:.=\-]?\s*(\d+(?:[.,/]\d+)?\s*(?:ml|kg|mg|g|l|m|muong(?:\s+(?:canh|ca phe))?)?)(?![a-z0-9])')
    values: dict[str, str] = {}
    for text in texts:
        text = unicodedata.normalize('NFC', text)
        for match in pattern.finditer(_fold(text)):
            # Keep the original unit spelling/case; compact OCR labels often
            # carry meaningful unit abbreviations. Do not rewrite to gold.
            value = text[match.start(1):match.end(1)].strip()
            values.setdefault(_fold(value).replace(' ', ''), value)
    return next(iter(values.values())) if len(values) == 1 else ''


def _semantic_answer_cue(evidence: EvidencePack, question: str) -> float:
    """Estimate whether evidence contains the requested answer semantics.

    This is deliberately corpus-agnostic: it uses the request description,
    broad answer types, and the evidence text, never a video/frame identity or
    benchmark answer. A distinctive description word such as ``cháo`` plus a
    compatible answer entity such as an animal species is therefore stronger
    than unrelated cooking speech that merely happens to exist near a frame.
    """
    evidence_text = _evidence_text(evidence)
    if not evidence_text:
        return 0.0
    question_folded = _fold(question)
    type_score = 0.0
    subject = quantity_subject(question)
    if subject:
        target = r'\s*'.join(re.escape(word) for word in subject.split())
        # Explicit entity/amount evidence can promote a deep retrieved video;
        # general cooking speech or an unrelated number cannot.
        if re.search(rf'(?<![a-z]){target}(?=\W|\d|$)[^\d;\n]{{0,24}}\d', _fold(evidence_text)):
            type_score = 1.0
    if _ANIMAL_QUESTION.search(question_folded):
        typed_answer = extract_typed_text_answer(question, evidence_text)
        if typed_answer:
            # A generic class such as "cá" is still useful evidence, but a
            # concrete species/ingredient name is substantially more
            # answer-bearing for unknown-X questions.
            type_score = 0.65 if typed_answer in {"Cá", "Động vật"} else 1.0

    description_tokens = _meaningful_tokens(evidence.query_text or "")
    evidence_tokens = _meaningful_tokens(evidence_text)
    overlap_count = len(description_tokens & evidence_tokens)
    overlap_score = min(1.0, overlap_count / 2.0)
    if type_score:
        return 0.70 * type_score + 0.30 * overlap_score
    return 0.30 * overlap_score


def score_answerability(
    candidate: SearchCandidate,
    evidence: EvidencePack,
    question: str,
    *,
    maximum_retrieval_score: float = 1.0,
) -> AnswerabilityScore:
    """Score whether a window can answer the question, not merely match it."""
    preferred = preferred_modalities(question)
    available = {
        "visual": bool(evidence.selected_frames),
        "asr": bool(evidence.asr_evidence or evidence.asr_texts),
        "ocr": bool(evidence.ocr_evidence or evidence.ocr_texts),
        "object": bool(evidence.object_evidence or evidence.object_labels),
        "metadata": bool(evidence.metadata_evidence),
    }
    modality_match = sum(1.0 for name in preferred if available.get(name, False)) / len(preferred)

    prefers_asr = "asr" in preferred
    transcript_answer_cue = 1.0 if available["asr"] and prefers_asr else (0.55 if available["asr"] else 0.0)

    visual_object_answer_cue = min(
        1.0,
        (0.50 if available["visual"] else 0.0)
        + (0.30 if available["object"] else 0.0)
        + (0.20 if available["ocr"] else 0.0),
    )

    temporal_signals = 0
    temporal_signals += int(bool(evidence.selected_frames))
    temporal_signals += int(bool(evidence.asr_evidence))
    temporal_signals += int(candidate.window_start_ms is not None and candidate.window_end_ms is not None)
    temporal_completeness = temporal_signals / 3.0
    semantic_answer_cue = _semantic_answer_cue(evidence, question)

    retrieval_relevance = _retrieval_relevance(candidate, maximum_retrieval_score)
    total = (
        0.24 * retrieval_relevance
        + 0.14 * modality_match
        + 0.10 * transcript_answer_cue
        + 0.14 * visual_object_answer_cue
        + 0.08 * temporal_completeness
        + 0.30 * semantic_answer_cue
    )
    relation = candidate.provenance.get("qa_relation_verification")
    if candidate.provenance.get("qa_relation_query_match") is True:
        total += 0.12
    if isinstance(relation, dict) and relation.get("available") is True:
        relation_score = relation.get("score")
        if isinstance(relation_score, (int, float)) and not isinstance(relation_score, bool):
            total += 0.18 * (max(0.0, min(1.0, float(relation_score))) - 0.5)
        if relation.get("contradiction") is True:
            total -= 0.30
    return AnswerabilityScore(
        retrieval_relevance=retrieval_relevance,
        modality_match=modality_match,
        transcript_answer_cue=transcript_answer_cue,
        visual_object_answer_cue=visual_object_answer_cue,
        temporal_completeness=temporal_completeness,
        semantic_answer_cue=semantic_answer_cue,
        total=max(0.0, min(1.0, total)),
    )


def rank_answerable_evidence(
    candidates: Sequence[SearchCandidate],
    evidence_packs: Sequence[EvidencePack],
    question: str,
) -> list[RankedEvidence]:
    if len(candidates) != len(evidence_packs):
        raise ValueError("candidates and evidence_packs must have equal length")
    numeric_scores = [
        float(value)
        for candidate in candidates
        for value in (candidate.score if candidate.score is not None else candidate.confidence,)
        if value is not None and not isinstance(value, bool) and float(value) > 0
    ]
    maximum = max(numeric_scores, default=1.0)
    ranked = [
        RankedEvidence(
            candidate=candidate,
            evidence=evidence,
            answerability=score_answerability(
                candidate, evidence, question, maximum_retrieval_score=maximum,
            ),
        )
        for candidate, evidence in zip(candidates, evidence_packs)
    ]
    ranked.sort(
        key=lambda item: (
            -item.answerability.total,
            item.candidate.rank,
            item.candidate.video_id,
            item.candidate.timestamp_ms,
            item.candidate.frame_id,
        )
    )
    return [
        replace(
            item,
            candidate=replace(
                item.candidate,
                rank=rank,
                provenance={
                    **item.candidate.provenance,
                    "qa_retrieval_rank": item.candidate.provenance.get("qa_retrieval_rank", item.candidate.rank),
                    "qa_answerability": item.answerability.to_json(),
                },
            ),
        )
        for rank, item in enumerate(ranked, start=1)
    ]


def select_answerable_shortlist(
    reranked: Sequence[RankedEvidence],
    retrieval_candidates: Sequence[SearchCandidate],
    limit: int,
) -> list[RankedEvidence]:
    """Keep open-vocabulary QA from discarding strong retrieval candidates.

    Evidence semantics may promote candidates, while a bounded reservation of
    distinct top-retrieval videos guarantees that an unseen answer type can
    still reach the open answer engine. One slot always remains available for
    evidence-driven promotion when ``limit`` is greater than one.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    # Evidence-rich windows from one unrelated video must not consume all
    # answer slots. Choose the best window of each video before duplicates.
    first_windows: list[RankedEvidence] = []
    other_windows: list[RankedEvidence] = []
    video_ids: set[str] = set()
    for item in reranked:
        if item.candidate.video_id in video_ids:
            other_windows.append(item)
        else:
            video_ids.add(item.candidate.video_id)
            first_windows.append(item)
    selected = (first_windows + other_windows)[:limit]
    if len(selected) <= 1:
        return selected

    reserve_count = min(3, len(selected) - 1)
    anchors: list[tuple[str, int]] = []
    seen_videos: set[str] = set()
    for candidate in retrieval_candidates:
        if candidate.video_id in seen_videos:
            continue
        seen_videos.add(candidate.video_id)
        anchors.append((candidate.video_id, candidate.frame_id))
        if len(anchors) >= reserve_count:
            break
    anchor_set = set(anchors)
    selected_ids = {
        (item.candidate.video_id, item.candidate.frame_id) for item in selected
    }
    for identity in anchors:
        if identity in selected_ids:
            continue
        anchor_item = next((
            item for item in reranked
            if (item.candidate.video_id, item.candidate.frame_id) == identity
        ), None)
        replacement_index = next((
            index for index in range(len(selected) - 1, -1, -1)
            if (selected[index].candidate.video_id, selected[index].candidate.frame_id)
            not in anchor_set
        ), None)
        if anchor_item is None or replacement_index is None:
            continue
        removed = selected[replacement_index]
        selected_ids.discard((removed.candidate.video_id, removed.candidate.frame_id))
        selected[replacement_index] = anchor_item
        selected_ids.add(identity)
    selected.sort(key=lambda item: (
        -item.answerability.total,
        item.candidate.rank,
        item.candidate.video_id,
        item.candidate.timestamp_ms,
    ))
    return selected
