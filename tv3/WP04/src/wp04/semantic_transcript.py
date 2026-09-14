"""Optional semantic ASR retrieval over a precomputed, versioned vector matrix."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retrieval import TextHit


@dataclass(frozen=True)
class SemanticTranscriptIndex:
    segment_ids: tuple[str, ...]
    vectors: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "SemanticTranscriptIndex":
        with np.load(path, allow_pickle=False) as payload:
            segment_ids = tuple(str(value) for value in payload["segment_ids"].tolist())
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(segment_ids) or not segment_ids:
            raise ValueError("semantic ASR index shape is invalid")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(vectors).all():
            raise ValueError("semantic ASR index contains invalid vectors")
        return cls(segment_ids, np.ascontiguousarray(vectors / norms, dtype=np.float32))

    def search(self, query_vector: np.ndarray, limit: int) -> list[TextHit]:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.vectors.shape[1] or not np.isfinite(vector).all():
            raise ValueError("semantic query vector dimension is invalid")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return []
        scores = self.vectors @ (vector / norm)
        order = np.lexsort((np.arange(len(scores)), -scores))[:limit]
        return [TextHit(self.segment_ids[int(index)], float(scores[int(index)])) for index in order]


class OpenAICompatibleEmbedder:
    """Small dependency-free adapter for local vLLM or an allowed cloud API."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 5.0) -> None:
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"model": self.model, "input": text}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError("embedding API returned malformed data")
        return np.asarray(data[0].get("embedding"), dtype=np.float32)


_CACHE: tuple[str, SemanticTranscriptIndex] | None = None


def semantic_search(query: str, limit: int) -> list[TextHit] | None:
    """Return None when semantic retrieval is unconfigured/unavailable."""
    global _CACHE
    index_path = os.environ.get("WP04_ASR_SEMANTIC_INDEX")
    base_url = os.environ.get("WP04_EMBEDDING_BASE_URL")
    model = os.environ.get("WP04_EMBEDDING_MODEL")
    if not (index_path and base_url and model):
        return None
    try:
        if _CACHE is None or _CACHE[0] != index_path:
            _CACHE = (index_path, SemanticTranscriptIndex.load(Path(index_path)))
        embedder = OpenAICompatibleEmbedder(
            base_url, model, os.environ.get("WP04_EMBEDDING_API_KEY", ""),
            float(os.environ.get("WP04_EMBEDDING_TIMEOUT_S", "5")),
        )
        return _CACHE[1].search(embedder.embed(query), limit)
    except (OSError, ValueError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
