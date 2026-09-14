"""Pinned MetaCLIP 2 adapter identity and worker entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..model_lock import ModelIdentity
from .common import run_worker
from .hf_clip import HuggingFaceClipAdapter

REVISION = "c139061af7b10fdb2e754b60d2b1182a3d5526c2"
MODEL_ID = "facebook/metaclip-2-worldwide-huge-quickgelu"
OFFICIAL_URL = f"https://huggingface.co/{MODEL_ID}"


class Adapter(HuggingFaceClipAdapter):
    @classmethod
    def identity(cls) -> ModelIdentity:
        return ModelIdentity("metaclip2", OFFICIAL_URL, REVISION)

    model_id = MODEL_ID
    revision = REVISION


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("request_path", type=Path)
    return run_worker(Adapter(), parser.parse_args(argv).request_path)


if __name__ == "__main__":
    raise SystemExit(main())
