from __future__ import annotations

import json
import io
import sys

import numpy as np

from wp03.cli import CliServices, _select_online_manifests, main
from wp03.corpus import load_corpus
from wp03.orchestrator import BuildRequest, build_model_artifacts
from pathlib import PurePosixPath
from tests.conftest import write_jsonl
from tests.test_corpus import record


class FakeEncoder:
    def encode_images(self, paths):
        return np.asarray([[1.0, 0.0] for _ in paths], dtype=np.float32)

    def encode_text(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_online_model_selection_preserves_requested_order(tmp_path) -> None:
    manifests = []
    for name in ("beit3", "bge_vl", "metaclip2"):
        path = tmp_path / f"{name}.json"
        path.touch()
        manifests.append(path)

    selected = _select_online_manifests(manifests, "bge_vl,metaclip2")

    assert [path.stem for path in selected] == ["bge_vl", "metaclip2"]


def fake_model_config(model_key: str) -> str:
    return f"""rows_per_shard: 1
models:
  {model_key}:
    model_id: test/{model_key}
    revision: rev
    tokenizer_revision: rev
    image_preprocess: {{image_size: 1}}
    text_preprocess: {{unicode_normalization: NFC}}
    query_template: '{{query}}'
    expected_dimension: 2
    dtype: float32
    fallback_dtype: null
    batch_size: 1
    timeout_seconds: 30
"""


def test_validate_rejects_traversal_path(data_root) -> None:
    assert main(["validate", "--data-root", str(data_root), "--frames", "../frames.jsonl"]) != 0


def test_search_prints_pipeline_envelope(data_root, tmp_path, capsys) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = tmp_path / "artifacts"
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 2), FakeEncoder())
    services = CliServices(encoders={"fake": FakeEncoder()})

    exit_code = main(
        ["search", "--artifact-root", str(artifact_root), "--query", "red car", "--top-k", "1"],
        services,
    )

    body = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert body["schema_version"] == "1.0.0"
    assert body["candidates"][0]["source"] == "visual"


def test_search_batch_prints_one_response_per_query(data_root, tmp_path, capsys) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = tmp_path / "artifacts"
    build_model_artifacts(BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 2), FakeEncoder())
    services = CliServices(encoders={"fake": FakeEncoder()})
    queries = json.dumps([
        {"query_id": "whole", "query_text": "red car", "event_index": None},
        {"query_id": "event-1", "query_text": "lands", "event_index": 0},
    ])

    exit_code = main(
        ["search-batch", "--artifact-root", str(artifact_root), "--queries-json", queries, "--top-k", "1"],
        services,
    )

    body = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [item["query_id"] for item in body["results"]] == ["whole", "event-1"]


def test_stdio_coordinator_serves_multiple_requests_in_one_process(
    data_root, tmp_path, capsys, monkeypatch,
) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)
    artifact_root = tmp_path / "artifacts"
    build_model_artifacts(
        BuildRequest("run", "fake", "rev", data_root, artifact_root, corpus, 2), FakeEncoder(),
    )
    requests = [
        {"operation": "search", "query_id": "q-1", "query_text": "red car", "top_k": 1},
        {"operation": "search", "query_id": "q-2", "query_text": "blue car", "top_k": 1},
    ]
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("".join(json.dumps(item) + "\n" for item in requests)),
    )

    exit_code = main(
        ["serve-stdio", "--artifact-root", str(artifact_root), "--top-k", "1"],
        CliServices(encoders={"fake": FakeEncoder()}),
    )

    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0
    assert [item["query_id"] for item in responses] == ["q-1", "q-2"]


def test_search_rejects_empty_query(tmp_path) -> None:
    assert main(["search", "--artifact-root", str(tmp_path), "--query", "   ", "--top-k", "1"]) != 0


def test_build_rejects_unsafe_run_id(data_root, tmp_path) -> None:
    config = tmp_path / "smoke.yaml"
    config.write_text("rows_per_shard: 2\nmodels: {}\n", encoding="utf-8")
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("workers: {}\n", encoding="utf-8")
    assert main(
        [
            "build", "--data-root", str(data_root), "--frames", "frames.jsonl", "--run-id", "../bad",
            "--config", str(config), "--runtime-root", str(tmp_path), "--runtime-profile", str(runtime),
            "--content-validation", "strict", "--code-version", "test-code",
        ]
    ) != 0


def test_build_requires_code_version_and_emits_build_summary(data_root, tmp_path, capsys) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    config = tmp_path / "smoke.yaml"
    config.write_text(fake_model_config("fake"), encoding="utf-8")
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("workers: {}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    exit_code = main(
        [
            "build", "--data-root", str(data_root), "--frames", "frames.jsonl", "--run-id", "run",
            "--config", str(config), "--runtime-root", str(tmp_path), "--runtime-profile", str(runtime),
            "--content-validation", "strict", "--artifact-root", str(artifact_root),
            "--code-version", "test-code",
        ],
        CliServices(encoders={"fake": FakeEncoder()}),
    )

    body = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert body["degraded"] is False
    assert (artifact_root / "reports" / "build-summary.json").is_file()


def test_build_returns_nonzero_when_every_model_fails(data_root, tmp_path, capsys) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    config = tmp_path / "smoke.yaml"
    config.write_text(fake_model_config("bad"), encoding="utf-8")
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("workers: {}\n", encoding="utf-8")

    class FailingEncoder:
        def encode_images(self, paths):
            raise RuntimeError("backend failed")

    exit_code = main(
        [
            "build", "--data-root", str(data_root), "--frames", "frames.jsonl", "--run-id", "run",
            "--config", str(config), "--runtime-root", str(tmp_path), "--runtime-profile", str(runtime),
            "--content-validation", "strict", "--artifact-root", str(tmp_path / "artifacts"),
            "--code-version", "test-code",
        ],
        CliServices(encoders={"bad": FailingEncoder()}),
    )

    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert exit_code != 0
