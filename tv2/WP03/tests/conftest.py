from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    keyframes = tmp_path / "keyframes" / "L21_V001"
    keyframes.mkdir(parents=True)
    (keyframes / "000003.jpg").write_bytes(b"image-three")
    (keyframes / "000042.jpg").write_bytes(b"image-forty-two")
    return tmp_path


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
