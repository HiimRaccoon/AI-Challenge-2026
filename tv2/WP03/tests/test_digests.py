from __future__ import annotations

from pathlib import Path, PurePosixPath

from wp03.corpus import load_corpus
from wp03.digests import ContentValidationMode, compute_corpus_digests, compute_record_digests
from tests.conftest import write_jsonl
from tests.test_corpus import record


def test_strict_digest_changes_when_image_bytes_change(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)

    before = compute_corpus_digests(corpus, data_root, ContentValidationMode.STRICT, "frames-digest", "tv1")
    (data_root / "keyframes/L21_V001/000003.jpg").write_bytes(b"changed-image")
    after = compute_corpus_digests(corpus, data_root, ContentValidationMode.STRICT, "frames-digest", "tv1")

    assert after.corpus_content_digest != before.corpus_content_digest


def test_fast_mode_records_fallback_when_tv1_manifest_is_absent(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)

    result = compute_corpus_digests(corpus, data_root, ContentValidationMode.FAST, "frames-digest", None)

    assert result.content_integrity_source == "frames_jsonl_fallback"


def test_parallel_strict_record_digests_match_sequential(data_root: Path) -> None:
    write_jsonl(data_root / "frames.jsonl", [record(3, "keyframes/L21_V001/000003.jpg")])
    corpus = load_corpus(data_root, PurePosixPath("frames.jsonl"), None)

    sequential = compute_record_digests(corpus, data_root, ContentValidationMode.STRICT)
    parallel = compute_record_digests(corpus, data_root, ContentValidationMode.STRICT, workers=2)

    assert parallel == sequential
