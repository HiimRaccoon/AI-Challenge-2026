"""Optional OpenAI-compatible query assistance with strict local fallback."""

from __future__ import annotations

import hashlib
import base64
import json
import mimetypes
import os
import re
import threading
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .qa_query_plan import build_qa_query_plan, relation_preserving_queries
from .query_enhancer import enhance_visual_query


_VALID_BRANCHES = {"visual", "ocr", "asr", "object", "metadata"}
_MAX_VISUAL_QUERY_WORDS = 60
_MAX_VISUAL_QUERY_VARIANTS = 5
_MAX_VARIANT_WORDS = 48
_SHARED_QUERY_CACHE: dict[str, object] = {}
_SHARED_QUERY_CACHE_LOCK = threading.Lock()
_STRONG_MOMENT_SPLIT = re.compile(
    r"\s*(?:[;\n]+\s*(?:[-*•]\s*)?|(?<=[.!?])\s+(?=\S))",
    re.IGNORECASE,
)
_TRANSITION_MOMENT_SPLIT = re.compile(
    r"\s*,?\s*(?:sau\s+đó|sau\s+do|tiếp\s+theo|tiep\s+theo|"
    r"cuối\s+cùng|cuoi\s+cung|chuyển\s+sang|chuyen\s+sang|"
    r"then|followed\s+by|afterwards?|next|finally)\s*",
    re.IGNORECASE,
)
_LEADING_TRANSITION = re.compile(
    r"^(?:sau\s+đó|sau\s+do|tiếp\s+theo|tiep\s+theo|"
    r"cuối\s+cùng|cuoi\s+cung|chuyển\s+sang|chuyen\s+sang|"
    r"then|followed\s+by|afterwards?|next|finally)\s*",
    re.IGNORECASE,
)
_TEMPORAL_ORDER_CUES = re.compile(
    r"\b(?:first|then|next|finally|afterwards?|followed\s+by|"
    r"lan\s+luot|sau\s+do|tiep\s+theo|cuoi\s+cung|"
    r"chuyen\s+sang|chuyen\s+canh|thu\s+nhat|thu\s+hai)\b",
    re.IGNORECASE,
)


def split_visual_moments(text: str) -> tuple[str, ...]:
    """Return bounded, distinct visible moments without model assistance.

    This deliberately understands formatting and temporal connectors rather
    than any benchmark-specific noun.  It is the reliable fallback when the
    optional translation model is slow or unavailable.
    """
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return ()
    strong_parts = _STRONG_MOMENT_SPLIT.split(cleaned)
    # Explicit punctuation already preserves compound moments such as
    # "second ingredient then colorful ingredients". Only fall back to
    # transition splitting when the prompt is a single unformatted sentence.
    parts = strong_parts if len(strong_parts) > 1 else _TRANSITION_MOMENT_SPLIT.split(cleaned)
    moments: list[str] = []
    for raw_value in parts:
        if raw_value.rstrip().endswith(":"):
            continue
        value = _LEADING_TRANSITION.sub("", raw_value)
        value = re.sub(r"^\s*(?:[-*•]|\d+[.)]|[Ee]\d+\s*:)\s*", "", value).strip(" ,.;:\n\t")
        value = re.sub(r"^(?:là|la|is)\s+", "", value, flags=re.IGNORECASE)
        if len(value.split()) < 8 and re.match(r"^(?:đây là|day la|this is)\b", value, re.IGNORECASE):
            continue
        if len(value.split()) < 3:
            continue
        if not any(_queries_are_near_duplicates(value, existing) for existing in moments):
            moments.append(value)
    return tuple(moments[: _MAX_VISUAL_QUERY_VARIANTS - 1])


def is_temporal_visual_query(text: str) -> bool:
    """Return true only for an explicitly ordered, multi-moment description.

    Semicolon-separated object lists are not temporal sequences. Requiring an
    order cue keeps the temporal reranker away from ordinary single-scene KIS
    queries while still covering Vietnamese and English transition language.
    """
    folded = "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")
    return len(split_visual_moments(text)) >= 2 and bool(_TEMPORAL_ORDER_CUES.search(folded))


def _bounded_visual_query(text: str) -> str:
    """Keep resident visual encoders inside their shared token envelope."""
    words = text.split()
    if len(words) <= _MAX_VISUAL_QUERY_WORDS:
        return text.strip()
    return " ".join(words[:_MAX_VISUAL_QUERY_WORDS])


def _bounded_variant(text: str) -> str:
    return " ".join(text.split()[:_MAX_VARIANT_WORDS]).strip()


def _atomic_moment_query(text: str) -> str:
    """Keep a distinctive action clause short enough for visual encoders.

    Deterministic concept expansion can produce a useful subject followed by
    an action and then redundant object aliases. For an atomic moment, retain
    the closest preceding subject plus the complete action phrase. The full
    description anchor still carries colors, setting, and co-occurrence.
    """
    prefix = "photo of "
    body = text.removeprefix(prefix)
    terms = [term.strip() for term in body.split(",") if term.strip()]
    action_index = next((
        index for index, term in enumerate(terms)
        if re.search(
            r"\b(?:playing|pulling|holding|giving|receiving|performing|"
            r"speaking|cutting|pouring|jumping|running|walking|driving|"
            r"wearing|dancing|singing)\b",
            term,
            re.IGNORECASE,
        )
    ), None)
    if action_index is None:
        return _bounded_variant(text)
    subject = terms[action_index - 1] if action_index > 0 else ""
    action = terms[action_index]
    return _bounded_variant(" ".join(value for value in (subject, action) if value))


def _usable_visual_translation(original: str, translated: object) -> str | None:
    """Accept only an actual English search sentence from the small assistant.

    Small local chat models sometimes answer the translation prompt with a
    category summary such as ``people, 1, objects, 0`` or simply echo the
    Vietnamese input.  Both are valid-looking JSON but destroy most retrieval
    evidence.  Reject those shapes and let the deterministic concept enhancer
    provide a conservative fallback.
    """
    if not isinstance(translated, str):
        return None
    query = translated.strip()
    if len(query.split()) < 3:
        return None
    folded = query.casefold()
    if any(marker in folded for marker in (
        "visible elements", "people, count", "objects, count",
        "people, 1", "objects, 0", "actions, 0", "scene, 1",
    )):
        return None
    original_has_non_ascii = any(ord(char) > 127 for char in original)
    translated_ascii_ratio = sum(ord(char) < 128 for char in query) / max(1, len(query))
    if original_has_non_ascii and translated_ascii_ratio < 0.95:
        return None
    # The assistant must preserve explicit high-value visual attributes. Small
    # local models occasionally produce fluent but destructive substitutions
    # (for example ``cháo`` -> noodle soup or ``màu cam`` -> camouflaged).
    # Reject only when the source contains an unambiguous phrase and the
    # translation contains none of its valid English realizations; the audited
    # deterministic enhancer then becomes the fallback.
    original_folded = "".join(
        char for char in unicodedata.normalize("NFD", original.casefold())
        if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")
    required_terms = (
        (r"\b(?:to chao|mon chao)\b", ("porridge", "congee")),
        (r"\bmau cam\b", ("orange",)),
        (r"\bmau den\b", ("black",)),
        (r"\bmau trang\b", ("white",)),
        (r"\bmau do\b", ("red",)),
        (r"\bmau vang\b", ("yellow",)),
        (r"\bmau xanh\b", ("blue", "green")),
    )
    for source_pattern, english_terms in required_terms:
        if re.search(source_pattern, original_folded) and not any(
            term in folded for term in english_terms
        ):
            return None
    return query


def _queries_are_near_duplicates(left: str, right: str, *, threshold: float = 0.82) -> bool:
    left_tokens = {token.strip(".,:;!?()[]{}\"'").casefold() for token in left.split()}
    right_tokens = {token.strip(".,:;!?()[]{}\"'").casefold() for token in right.split()}
    left_tokens.discard("")
    right_tokens.discard("")
    if not left_tokens or not right_tokens:
        return left.strip().casefold() == right.strip().casefold()
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens)) >= threshold


@dataclass
class ExternalQueryAssist:
    enabled: bool
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 2.0
    _cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def from_environment(cls) -> "ExternalQueryAssist":
        return cls(
            enabled=os.environ.get("TV4_EXTERNAL_ASSIST_ENABLED", "1") == "1",
            base_url=os.environ.get("TV4_EXTERNAL_ASSIST_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/"),
            model=os.environ.get("TV4_EXTERNAL_ASSIST_MODEL", "qwen2.5:3b-instruct"),
            api_key=os.environ.get("TV4_EXTERNAL_ASSIST_API_KEY", "ollama"),
            timeout=float(os.environ.get("TV4_EXTERNAL_ASSIST_TIMEOUT_S", "2")),
        )

    def _chat_json(self, purpose: str, system: str, user: str) -> object | None:
        if not (self.enabled and self.base_url and self.model):
            return None
        key = hashlib.sha256(f"{purpose}\0{self.model}\0{user}".encode("utf-8")).hexdigest()
        with _SHARED_QUERY_CACHE_LOCK:
            if key in _SHARED_QUERY_CACHE:
                return _SHARED_QUERY_CACHE[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "temperature": 0,
                "seed": 0,
                "max_tokens": 1024 if purpose == "translate_sequence" else 256,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (OSError, TimeoutError, urllib.error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None
        with self._lock:
            if len(self._cache) >= 512:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result
        with _SHARED_QUERY_CACHE_LOCK:
            if len(_SHARED_QUERY_CACHE) >= 512:
                _SHARED_QUERY_CACHE.pop(next(iter(_SHARED_QUERY_CACHE)))
            _SHARED_QUERY_CACHE[key] = result
        return result

    def visual_query(self, original: str, *, include_concepts: bool = False) -> str:
        result = self._chat_json(
            "translate",
            "Translate the Vietnamese text faithfully into concise English for image retrieval. "
            "Keep every explicitly visible subject, object, color, action, direction, number, "
            "camera viewpoint, and temporal order. Do not summarize, count categories, label "
            "fields, or add facts. Return JSON with one string field named query only.",
            original,
        )
        query = result.get("query") if isinstance(result, dict) else None
        usable = _usable_visual_translation(original, query)
        concepts = enhance_visual_query(original, allow_online_translation=False)
        if usable is None:
            usable = concepts
        elif include_concepts and "\nVisual keywords: " in concepts:
            # Translation preserves literal detail; the audited concept map
            # adds visual synonyms and symbolic scene semantics that CLIP-like
            # encoders often recognize better (for example tug-of-war as a
            # competition). Both are derived from the user's words.
            concept_terms = concepts.split("\nVisual keywords: ", 1)[1].strip()
            if concept_terms and concept_terms.casefold() not in usable.casefold():
                usable = f"{usable}, {concept_terms}"
        return _bounded_visual_query(usable)

    def visual_queries(
        self, original: str, *, deterministic_anchor: bool = False,
    ) -> tuple[str, ...]:
        """Build a bounded visual query set with at most one model call.

        Ordered descriptions use deterministic per-moment expansions after a
        single faithful anchor translation. This removes the previous chain of
        one LLM request per clause and, importantly, prevents a generic food or
        scene bundle from being mistaken for a temporal moment.
        """
        anchor = (
            _bounded_visual_query(enhance_visual_query(original, allow_online_translation=False))
            if deterministic_anchor
            else self.visual_query(original)
        )
        variants: list[str] = [anchor]
        moments = split_visual_moments(original)
        raw_concept_bundle = enhance_visual_query(
            original, allow_online_translation=False,
        )
        concept_terms = [
            term.strip()
            for term in raw_concept_bundle.partition("\nVisual keywords: ")[2].split(",")
            if term.strip()
        ]
        # Keep a compact subject/object signature. Camera vocabulary remains
        # in the faithful anchor and moment clauses; allowing it to dominate
        # this query makes visually distinctive object co-occurrence generic.
        subject_terms = [
            term for term in concept_terms
            if not re.search(r"\b(?:camera|recording|shot|view)\b", term, re.IGNORECASE)
        ]
        distinct_subject_terms: list[str] = []
        for term in subject_terms:
            term_tokens = set(term.casefold().split())
            if any(term_tokens <= set(existing.casefold().split()) for existing in distinct_subject_terms):
                continue
            distinct_subject_terms.append(term)
        subject_terms = distinct_subject_terms
        if len(subject_terms) >= 2:
            raw_concept_bundle = "photo of " + ", ".join(subject_terms)
        concept_bundle = _bounded_variant(raw_concept_bundle)
        if is_temporal_visual_query(original):
            # Position one is a video-level locator and is deliberately not a
            # temporal step. Keeping its role stable lets the chain scorer use
            # only positions two onward while preserving the strong recall of
            # object co-occurrence queries.
            variants.append(concept_bundle or anchor)
            for clause in moments:
                bounded = _bounded_variant(enhance_visual_query(
                    clause, allow_online_translation=False,
                ))
                if bounded and not any(
                    _queries_are_near_duplicates(bounded, item) for item in variants[2:]
                ):
                    variants.append(bounded)
                if len(variants) >= _MAX_VISUAL_QUERY_VARIANTS:
                    break
            return tuple(variants)

        # A translated long sentence is good for relations but often weak for
        # CLIP-style retrieval. Ordinary single-scene queries retain one
        # concise deterministic co-occurrence query and an optional action
        # anchor, but never enter the temporal-chain path.
        if (
            concept_bundle
            and concept_bundle.casefold() != anchor.casefold()
            and not _queries_are_near_duplicates(concept_bundle, anchor)
        ):
            variants.append(concept_bundle)
        # Small translation models sometimes map a precise action to a nearby
        # generic pose (for example a rope contest to wrestling). Put audited
        # action concepts before the faithful anchor so they survive token
        # truncation while the anchor still supplies subjects, colors and
        # co-occurrence. The terms come only from the deterministic concept
        # map, never from corpus IDs or expected answers.
        action_terms = [
            term for term in concept_terms
            if re.search(
                r"\b(?:playing|pulling|holding|giving|receiving|performing|"
                r"speaking|cutting|pouring|jumping|running|walking|driving|"
                r"wearing|dancing|singing)\b",
                term,
                re.IGNORECASE,
            )
        ]
        if action_terms:
            action_anchor = _bounded_variant(
                f"{', '.join(action_terms)}. {anchor}",
            )
            if action_anchor and all(
                action_anchor.casefold() != item.casefold() for item in variants
            ):
                variants.append(action_anchor)
        # Unordered multi-panel or multi-object descriptions (for example two
        # illustrations on one lecture slide) still need independent recall
        # queries. They participate in video consensus but never in temporal
        # chain scoring because no ordering cue was present.
        moment_variants: list[str] = []
        for clause in moments:
            if len(variants) >= _MAX_VISUAL_QUERY_VARIANTS:
                break
            bounded = _atomic_moment_query(enhance_visual_query(
                clause, allow_online_translation=False,
            ))
            if bounded and not any(
                _queries_are_near_duplicates(bounded, item) for item in moment_variants
            ):
                variants.append(bounded)
                moment_variants.append(bounded)
        return tuple(variants[:_MAX_VISUAL_QUERY_VARIANTS])

    def trake_translations(self, description: str, events: tuple[str, ...]) -> tuple[str | None, ...]:
        """Translate a complete sequence in one bounded call, preserving slots."""
        originals = (description, *events)
        result = self._chat_json(
            "translate_sequence",
            "Translate each input string into faithful English for video retrieval. "
            "Keep every subject, action, count, ordinal, negation, beginning/completion "
            "condition and temporal order. Do not summarize or reorder entries. "
            'Return JSON only with a queries array of exactly the same length.',
            json.dumps({"queries": originals}, ensure_ascii=False),
        )
        queries = result.get("queries") if isinstance(result, dict) else None
        if not isinstance(queries, list) or len(queries) != len(originals):
            return tuple(None for _ in originals)
        return tuple(_usable_visual_translation(original, translated) for original, translated in zip(originals, queries))

    def vqa_visual_queries(self, description: str, question: str) -> tuple[str, ...]:
        """Add answer-source queries without detaching a display from its device.

        Description variants continue to identify the video.  The deterministic
        relation query only improves localization inside that video and is
        intentionally not sent to OCR/ASR lexical branches.
        """
        plan = build_qa_query_plan(description, question)
        relation_queries = relation_preserving_queries(description, question)
        # QA must be repeatable: a small translator timing out or replacing a
        # rare noun with a related one must not change the retrieved video.
        # KIS continues to use ``visual_queries`` with its default translated
        # anchor; this deterministic option is scoped to /vqa/answer.
        variants = list(self.visual_queries(description, deterministic_anchor=True))
        if (
            plan.answer_source != "device_display"
            and is_temporal_visual_query(description)
            and len(variants) >= 3
        ):
            # The compact opening moment usually carries the stable scene
            # identity (subject, clothing, setting) while a long all-moments
            # bundle is dominated by generic actions. Make that first moment
            # the QA retrieval anchor and retain the complete description as
            # supporting variants. This is structural, not topic vocabulary.
            # ``variants[1]`` is the same deterministic concept bundle as the
            # full anchor in this mode, so do not spend another visual-model
            # pass on it. Four distinct QA queries retain scene and actions.
            variants = [variants[2], variants[0], *variants[3:]][:4]
        if relation_queries and plan.answer_source == "device_display":
            # VQA retrieval keeps the complete event anchor plus the strict
            # answer-source relation. Extra moment queries remain useful for
            # KIS, but letting them vote equally here can drown the rare device
            # frame among many generic subject/action matches.
            variants = variants[:1]
        elif relation_queries:
            # Chart and bounded temporal questions benefit from independent
            # description moments, but reserve the last slot for the generic
            # answer-source relation instead of silently truncating it.
            variants = variants[: _MAX_VISUAL_QUERY_VARIANTS - 1]
        for query in relation_queries:
            bounded = _bounded_variant(query)
            if bounded and not any(_queries_are_near_duplicates(bounded, item) for item in variants):
                variants.append(bounded)
        return tuple(variants[:_MAX_VISUAL_QUERY_VARIANTS])

    def branches(self, original: str) -> tuple[str, ...]:
        result = self._chat_json(
            "router",
            "Select useful video retrieval modalities. Return JSON {\"branches\":[...]} using only visual, ocr, asr, object, metadata. ASR means spoken content; OCR means visible writing. Do not return any other value.",
            original,
        )
        raw = result.get("branches") if isinstance(result, dict) else None
        if not isinstance(raw, list):
            return ()
        return tuple(dict.fromkeys(value for value in raw if isinstance(value, str) and value in _VALID_BRANCHES))


@dataclass
class ExternalVlmVerifier:
    enabled: bool
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 8.0
    max_images: int = 10

    @classmethod
    def from_environment(cls) -> "ExternalVlmVerifier":
        return cls(
            enabled=os.environ.get("TV4_EXTERNAL_VLM_ENABLED", "0") == "1",
            base_url=os.environ.get("TV4_EXTERNAL_VLM_BASE_URL", os.environ.get("TV4_EXTERNAL_ASSIST_BASE_URL", "")).rstrip("/"),
            model=os.environ.get("TV4_EXTERNAL_VLM_MODEL", os.environ.get("TV4_EXTERNAL_ASSIST_MODEL", "")),
            api_key=os.environ.get("TV4_EXTERNAL_VLM_API_KEY", os.environ.get("TV4_EXTERNAL_ASSIST_API_KEY", "")),
            timeout=float(os.environ.get("TV4_EXTERNAL_VLM_TIMEOUT_S", "8")),
            max_images=max(1, min(20, int(os.environ.get("TV4_EXTERNAL_VLM_TOP_K", "10")))),
        )

    def rank(self, query: str, image_paths: list[Path]) -> list[tuple[int, float]] | None:
        """Return zero-based input indices only; the model cannot invent a frame."""
        if not (self.enabled and self.base_url and self.model and image_paths):
            return None
        content: list[dict[str, object]] = [{
            "type": "text",
            "text": (
                "Rank only the numbered candidate images that visibly satisfy this video retrieval query. "
                "Return JSON {\"selected\":[{\"number\":1,\"score\":95}]} with 1-based numbers, score 0..100. "
                "Do not infer facts not visible. Query: " + query
            ),
        }]
        for number, path in enumerate(image_paths[: self.max_images], start=1):
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.extend((
                {"type": "text", "text": f"Candidate {number}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ))
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model, "temperature": 0, "max_tokens": 256,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": content}],
            }).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            parsed = json.loads(payload["choices"][0]["message"]["content"])
            selected = parsed.get("selected", [])
        except (OSError, TimeoutError, urllib.error.URLError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None
        ranked: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in selected if isinstance(selected, list) else ():
            if not isinstance(item, dict):
                continue
            number, score = item.get("number"), item.get("score")
            if isinstance(number, int) and 1 <= number <= min(len(image_paths), self.max_images) and number - 1 not in seen:
                seen.add(number - 1)
                ranked.append((number - 1, max(0.0, min(100.0, float(score or 0.0)))))
        return sorted(ranked, key=lambda item: (-item[1], item[0]))
