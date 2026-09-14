"""Deterministic text construction and token-budget checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from .contracts import FeedbackEvent, FeedbackValidationError


def build_feedback_template(original_query: str, events: Sequence[FeedbackEvent]) -> str:
    if not isinstance(original_query, str) or not original_query.strip():
        raise FeedbackValidationError("original query must be non-empty")
    lines = [f"Original query: {original_query}"]
    lines.extend(f"Refinement {index}: {event.feedback_text}" for index, event in enumerate(events, start=1))
    return "\n".join(lines)


def validate_token_budget(template: str, *, token_counter: Callable[[str], int], limit: int = 64) -> None:
    if token_counter(template) > limit:
        raise FeedbackValidationError(f"feedback template exceeds {limit} tokens")


class Beit3TokenCounter:
    """Exact tokenizer family used by WP03's BEiT-3 worker, loaded lazily."""

    def __init__(self, sentencepiece_path: Path) -> None:
        self._path = sentencepiece_path
        self._tokenizer = None

    def __call__(self, template: str) -> int:
        if self._tokenizer is None:
            from transformers import XLMRobertaTokenizer

            self._tokenizer = XLMRobertaTokenizer(str(self._path))
        encoded = self._tokenizer(template, truncation=False, add_special_tokens=True)
        return len(encoded["input_ids"])
