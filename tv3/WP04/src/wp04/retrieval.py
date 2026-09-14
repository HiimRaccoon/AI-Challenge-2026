from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log
from typing import Protocol

from .normalization import normalize_text
from .contracts import ObjectDetection, SearchCandidate


@dataclass(frozen=True, slots=True)
class TextHit:
    document_id: str
    score: float


@dataclass(frozen=True, slots=True)
class ObjectMatch:
    evidence_refs: tuple[str, ...]
    labels: tuple[str, ...]
    suggested_boost: float


class TextIndex(Protocol):
    def search(self, query: str, limit: int) -> list[TextHit]: ...


class LocalTextIndex:
    def __init__(self, documents: dict[str, str]):
        self._documents = {key: normalize_text(value) for key, value in documents.items()}
        self._tokens = {
            key: normalized.without_diacritics.split() for key, normalized in self._documents.items()
        }
        self._document_frequency: dict[str, int] = {}
        for tokens in self._tokens.values():
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens.values()) / len(self._tokens)
            if self._tokens else 0.0
        )

    @classmethod
    def from_documents(cls, documents: list[tuple[str, str]]) -> "LocalTextIndex":
        return cls(dict(documents))

    def search(self, query: str, limit: int) -> list[TextHit]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        normalized_query = normalize_text(query)
        normalized = normalized_query.without_diacritics
        query_tokens = normalized.split()
        if not query_tokens:
            return []
        query_terms = set(query_tokens)
        scored = []
        for document_id, text in self._documents.items():
            tokens = self._tokens[document_id]
            # A fuzzy-only match is useful for a typo in an otherwise matching
            # phrase (``banh mii`` still shares ``banh`` with ``banh mi``), but
            # it must not turn unrelated records into search results.  In the
            # latter case SequenceMatcher was effectively ranking every ASR/OCR
            # row and caused a VQA event search to surface arbitrary videos.
            if not query_terms.intersection(tokens):
                continue
            exact = float(normalized == text.without_diacritics)
            similarity = SequenceMatcher(None, normalized, text.without_diacritics).ratio()
            bm25 = self._bm25(tokens, query_tokens)
            ngram = self._ngram_jaccard(normalized, text.without_diacritics)
            scored.append(TextHit(document_id, exact * 4.0 + bm25 + similarity + ngram))
        return sorted(scored, key=lambda hit: (-hit.score, hit.document_id))[:limit]

    def _bm25(self, document_tokens: list[str], query_tokens: list[str]) -> float:
        if not document_tokens or not query_tokens or not self._average_length:
            return 0.0
        score = 0.0
        total_documents = len(self._tokens)
        for term in set(query_tokens):
            frequency = document_tokens.count(term)
            if not frequency:
                continue
            document_frequency = self._document_frequency.get(term, 0)
            inverse_frequency = log(1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            numerator = frequency * 2.2
            denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * len(document_tokens) / self._average_length)
            score += inverse_frequency * numerator / denominator
        return score

    @staticmethod
    def _ngram_jaccard(left: str, right: str, width: int = 3) -> float:
        def grams(value: str) -> set[str]:
            padded = f" {value} "
            return {padded[index:index + width] for index in range(max(0, len(padded) - width + 1))}

        left_grams, right_grams = grams(left), grams(right)
        if not left_grams or not right_grams:
            return 0.0
        return len(left_grams & right_grams) / len(left_grams | right_grams)


class ElasticTextIndex:
    """Optional backend that preserves local retrieval when Elasticsearch is unavailable."""

    def __init__(self, client: object, local: TextIndex) -> None:
        self._client = client
        self._local = local

    def search(self, query: str, limit: int) -> list[TextHit]:
        try:
            search = getattr(self._client, "search")
            hits = search(query, limit)
            return [
                hit if isinstance(hit, TextHit) else TextHit(hit["document_id"], float(hit["score"]))
                for hit in hits
            ][:limit]
        except Exception:
            return self._local.search(query, limit)


def object_match(
    candidate: SearchCandidate, detections: list[ObjectDetection], requested_labels: set[str],
) -> ObjectMatch:
    """Return optional object evidence; TV4 alone decides whether to apply its boost."""
    requested = {label.lower() for label in requested_labels}
    matching = sorted(
        (
            detection for detection in detections
            if detection.video_id == candidate.video_id
            and detection.frame_id == candidate.frame_id
            and detection.label.lower() in requested
        ),
        key=lambda item: (-item.confidence, item.evidence_id),
    )
    return ObjectMatch(
        evidence_refs=tuple(item.evidence_id for item in matching),
        labels=tuple(item.label for item in matching),
        suggested_boost=min(0.25, max((item.confidence * 0.25 for item in matching), default=0.0)),
    )
