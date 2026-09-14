"""Independent, conservative verification for QA answer hypotheses.

The answer engine may propose an answer and emit its own boolean verdict.  This
module does not call that engine.  It checks the proposal against the structured
OCR, ASR and object evidence already owned by TV4 and exposes calibrated support
components instead of promoting the producer verdict to ground truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata

from .contracts import EvidencePack
from .visual_verifier import VisualVerification
from .qa_reranker import extract_typed_text_answer
from .qa_query_plan import question_prefers_visual_text, quantity_subject


def _fold(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


_STOPWORDS = {
    "a", "an", "the", "la", "cua", "va", "thi", "duoc", "nay", "kia",
    "mot", "nhung", "nhung", "that", "do", "there", "is", "are", "was",
}
_YES = {"yes", "true", "co", "dung"}
_NO = {"no", "false", "khong", "not", "never"}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _fold(text))
        if len(token) > 1 and token not in _STOPWORDS
    }


def _support(answer: str, texts: list[str]) -> float:
    answer_folded = _fold(answer).strip()
    if not answer_folded or not texts:
        return 0.0
    answer_tokens = _tokens(answer)
    best = 0.0
    for text in texts:
        folded = _fold(text)
        if re.search(rf"(?<!\w){re.escape(answer_folded)}(?!\w)", folded):
            return 1.0
        if answer_tokens:
            best = max(best, len(answer_tokens & _tokens(text)) / len(answer_tokens))
    return best


def _answers_question(question: str, answer: str, texts: list[str]) -> bool:
    """Require requested type and local context, not a matching watermark.

    This is a conservative text check. Questions whose relation cannot be
    established lexically need the independent visual verifier or review.
    """
    folded_question = _fold(question)
    asks_identity = bool(re.search(r"\b(ten|nao|who|which|name)\b", folded_question))
    answer_tokens = _tokens(answer)
    if asks_identity and answer_tokens and answer_tokens <= _tokens(question):
        # "Which school?" -> "school" names the requested type, not an
        # entity. Leave explicit offered alternatives to visual verification.
        return False
    numeric = bool(re.search(r"\b(bao nhieu|may|how many|how much|what number)\b", folded_question))
    if numeric and not re.search(r"\d|\b(mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi|one|two|three|four|five|six|seven|eight|nine|ten)\b", _fold(answer)):
        return False
    subject = quantity_subject(question)
    if numeric and subject:
        # Bind the quantity to the requested ingredient/entity, not another
        # quantity elsewhere in the same recipe or a generic word like 'used'.
        target_words = r'\s*'.join(re.escape(word) for word in subject.split())
        target = re.compile(rf"(?<![a-z]){target_words}(?=\W|\d|$)")
        answer_pattern = re.compile(rf"(?<!\w){re.escape(_fold(answer).strip())}(?!\w)")
        for text in texts:
            folded = _fold(text)
            for match in target.finditer(folded):
                after = folded[match.end():match.end() + 80]
                before = folded[max(0, match.start() - 80):match.start()]
                for segment, reverse in ((after, False), (before, True)):
                    answers = list(answer_pattern.finditer(segment))
                    for found in answers:
                        gap = segment[found.end():] if reverse else segment[:found.start()]
                        # Another amount or list separator indicates a different
                        # entity/value pair (soy sauce 1M; cooking oil 2M).
                        if not re.search(r"\d|[;\n]", gap) and len(gap.strip()) <= 24:
                            return True
        return False
    if any(_fold(extract_typed_text_answer(question, text)) == _fold(answer) for text in texts):
        return True
    ignored = _STOPWORDS | {
        "gi", "nao", "ten", "hoi", "hay", "cho", "biet", "bao", "nhieu", "may",
        "nguoi", "duoc", "nay", "trong", "tren", "nhac", "den", "gioi", "thieu",
        "what", "which", "who", "where", "how", "many", "much", "does", "say",
        "name", "this", "that", "shown", "please", "tell",
    }
    anchors = _tokens(question) - ignored - answer_tokens
    pattern = re.compile(rf"(?<!\w){re.escape(_fold(answer).strip())}(?!\w)")
    for text in texts:
        folded = _fold(text)
        for match in pattern.finditer(folded):
            context = folded[max(0, match.start() - 120):match.end() + 120]
            if anchors & _tokens(context):
                return True
    return False


@dataclass(frozen=True)
class QaVerification:
    supported: bool
    visual_support: float
    audio_text_support: float
    ocr_support: float
    producer_support: float
    contradiction: bool
    final_support_score: float
    manual_required: bool
    visual_verifier_available: bool = False
    visual_frame_ids: tuple[int, ...] = ()
    visual_verifier_reason: str | None = None
    reasons: tuple[str, ...] = ()
    scene_supported: bool | None = None
    scene_score: float | None = None
    scene_verifier_reason: str | None = None

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "visual_support", "audio_text_support", "ocr_support",
            "producer_support", "final_support_score",
        ):
            payload[key] = round(float(payload[key]), 6)
        payload["visual_frame_ids"] = list(self.visual_frame_ids)
        payload["reasons"] = list(self.reasons)
        return payload


def verify_answer(
    question: str,
    answer: str,
    evidence: EvidencePack,
    *,
    producer_supported: bool = False,
    visual_verification: VisualVerification | None = None,
    description: str = "",
    scene_verification: VisualVerification | None = None,
) -> QaVerification:
    """Verify a proposal without generating text or deriving frame identity."""
    clean_answer = answer.strip()
    if not clean_answer:
        return QaVerification(
            supported=False,
            visual_support=0.0,
            audio_text_support=0.0,
            ocr_support=0.0,
            producer_support=0.0,
            contradiction=False,
            final_support_score=0.0,
            manual_required=True,
            visual_verifier_available=bool(visual_verification and visual_verification.available),
            visual_frame_ids=visual_verification.frame_ids if visual_verification else (),
            visual_verifier_reason=visual_verification.reason if visual_verification else None,
            reasons=("empty_answer",),
        )

    ocr_texts = [item.raw_text for item in evidence.ocr_evidence] + list(evidence.ocr_texts)
    asr_texts = (
        [item.text for item in evidence.asr_evidence]
        + [
            str(row.get("text", ""))
            for item in evidence.asr_evidence
            for row in item.context
            if isinstance(row, dict)
        ]
        + list(evidence.asr_texts)
    )
    object_texts = [
        item.canonical_label or item.label for item in evidence.object_evidence
    ] + list(evidence.object_labels)

    ocr_support = _support(clean_answer, ocr_texts)
    audio_support = _support(clean_answer, asr_texts)
    object_support = _support(clean_answer, object_texts)
    # A canonical image proves that visual evidence exists, but not that an
    # arbitrary text answer is visible in it. Keep that signal deliberately low.
    fallback_visual_support = max(object_support, 0.15 if evidence.selected_frames else 0.0)
    visual_available = bool(
        visual_verification
        and visual_verification.available
        and visual_verification.score is not None
    )
    visual_support = max(
        object_support,
        float(visual_verification.score) if visual_available and visual_verification else fallback_visual_support,
    )
    producer_support = 1.0 if producer_supported else 0.0

    answer_tokens = _tokens(clean_answer)
    evidence_tokens = _tokens(" ".join((*ocr_texts, *asr_texts, *object_texts)))
    answer_is_yes = bool(answer_tokens & _YES) and not bool(answer_tokens & _NO)
    answer_is_no = bool(answer_tokens & _NO) and not bool(answer_tokens & _YES)
    text_contradiction = (answer_is_yes and bool(evidence_tokens & _NO)) or (
        answer_is_no and bool(evidence_tokens & _YES)
    )
    contradiction = text_contradiction or bool(
        visual_available and visual_verification and visual_verification.contradiction
    )

    supports = sorted((visual_support, audio_support, ocr_support), reverse=True)
    final_score = 0.70 * supports[0] + 0.20 * supports[1] + 0.10 * producer_support
    if contradiction:
        final_score *= 0.25
    final_score = max(0.0, min(1.0, final_score))
    supported = final_score >= 0.55 and not contradiction
    # Spoken words cannot establish what lettering a sign/display shows.
    # Such questions need OCR context or independent image verification.
    question_texts = ocr_texts if question_prefers_visual_text(question) else ocr_texts + asr_texts
    question_supported = _answers_question(question, clean_answer, question_texts)
    visual_relation_supported = bool(visual_available and visual_verification and float(visual_verification.score) >= 0.55 and not visual_verification.contradiction)
    if not question_supported and not visual_relation_supported:
        supported = False
        final_score = min(final_score, 0.49)
    dense = evidence.provenance.get("dense_ocr")
    dense_relation = evidence.provenance.get("dense_ocr_relation_verification")
    dense_relation_verified = bool(
        isinstance(dense_relation, dict)
        and dense_relation.get("available") is True
        and dense_relation.get("contradiction") is not True
        and isinstance(dense_relation.get("score"), (int, float))
        and not isinstance(dense_relation.get("score"), bool)
        and float(dense_relation["score"]) >= 0.55
    )
    if isinstance(dense, dict) and dense.get("available") is True and not dense_relation_verified:
        supported = False

    reasons: list[str] = []
    if not question_supported and not visual_relation_supported:
        reasons.append("question_relation_unproven")
    if text_contradiction:
        reasons.append("explicit_text_contradiction")
    if visual_available and visual_verification and visual_verification.contradiction:
        reasons.append("visual_verifier_contradiction")
    if supports[0] == 0.0:
        reasons.append("no_answer_bearing_evidence")
    elif not supported:
        reasons.append("support_below_threshold")
    if isinstance(dense, dict) and dense.get("available") is True and not dense_relation_verified:
        reasons.append("device_relation_unverified")
    if producer_supported and not supported:
        reasons.append("producer_verdict_not_independently_corroborated")

    scene_supported = None
    if description.strip():
        scene_supported = bool(
            scene_verification and scene_verification.available
            and scene_verification.score is not None and scene_verification.score >= 0.7
            and not scene_verification.contradiction
        )
        if not scene_supported:
            supported = False
            final_score = min(final_score, 0.49)
            reasons.append('scene_mismatch' if scene_verification and scene_verification.contradiction else 'scene_unverified')
        if scene_verification and scene_verification.contradiction:
            contradiction = True
            final_score = min(final_score, 0.2)

    return QaVerification(
        supported=supported,
        visual_support=visual_support,
        audio_text_support=audio_support,
        ocr_support=ocr_support,
        producer_support=producer_support,
        contradiction=contradiction,
        final_support_score=final_score,
        manual_required=not supported,
        visual_verifier_available=visual_available,
        visual_frame_ids=visual_verification.frame_ids if visual_verification else (),
        visual_verifier_reason=visual_verification.reason if visual_verification else None,
        reasons=tuple(reasons),
        scene_supported=scene_supported,
        scene_score=scene_verification.score if scene_verification else None,
        scene_verifier_reason=scene_verification.reason if scene_verification else None,
    )
