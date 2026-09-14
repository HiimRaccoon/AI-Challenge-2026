"""Small, dependency-free WP04 retrieval quality and operational report."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable, Mapping

from .contracts import SearchCandidate


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left_value != right_value)))
        previous = current
    return previous[-1]


class BenchmarkRunner:
    def __init__(self, preprocess_run_id: str, wp04_artifact_set_id: str) -> None:
        self.preprocess_run_id = preprocess_run_id
        self.wp04_artifact_set_id = wp04_artifact_set_id

    def run(
        self, results: Mapping[str, Iterable[SearchCandidate]], relevant: Mapping[str, set[tuple[str, int]]],
        *, artifact_counts: Mapping[str, int] | None = None, error_categories: Mapping[str, int] | None = None,
        references: Mapping[str, str] | None = None, hypotheses: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        started = perf_counter()
        materialized = {query_id: list(candidates) for query_id, candidates in results.items()}
        total = len(relevant)

        def hit_at(limit: int) -> float:
            if not total:
                return 0.0
            return sum(
                any((candidate.video_id, candidate.frame_id) in relevant.get(query_id, set()) for candidate in candidates[:limit])
                for query_id, candidates in materialized.items()
            ) / total

        elapsed_ms = (perf_counter() - started) * 1000
        report: dict[str, object] = {
            "preprocess_run_id": self.preprocess_run_id,
            "wp04_artifact_set_id": self.wp04_artifact_set_id,
            "hit_at_1": hit_at(1), "hit_at_5": hit_at(5), "hit_at_20": hit_at(20),
            "latency_ms": elapsed_ms, "throughput_queries_per_second": total / max(elapsed_ms / 1000, 1e-9),
            "artifact_counts": dict(artifact_counts or {}), "error_categories": dict(error_categories or {}),
            "cer": None, "wer": None,
        }
        if references is not None and hypotheses is not None:
            shared = sorted(set(references) & set(hypotheses))
            if shared:
                character_error = sum(_distance(list(hypotheses[key]), list(references[key])) for key in shared)
                character_total = sum(max(1, len(references[key])) for key in shared)
                word_error = sum(_distance(hypotheses[key].split(), references[key].split()) for key in shared)
                word_total = sum(max(1, len(references[key].split())) for key in shared)
                report["cer"] = character_error / character_total
                report["wer"] = word_error / word_total
        return report
