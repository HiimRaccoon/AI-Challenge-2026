"""Pinned BGE-VL adapter identity and worker entry point."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from ..model_lock import ModelIdentity
from .common import run_worker

REVISION = "40fb48217f521df22a2a5bf15edd52ed1146ef05"
MODEL_ID = "BAAI/BGE-VL-large"
OFFICIAL_URL = f"https://huggingface.co/{MODEL_ID}"


class Adapter:
    def __init__(self) -> None:
        self._model = None
        self._dtype = "float32"

    def configure(self, *, dtype: str) -> None:
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported worker dtype")
        self._dtype = dtype

    @classmethod
    def identity(cls) -> ModelIdentity:
        return ModelIdentity("bge_vl", OFFICIAL_URL, REVISION)

    def _load(self):
        if self._model is None:
            local_files_only = os.environ.get("WP03_HF_LOCAL_ONLY", "1") != "0"
            if local_files_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModel

            if not torch.cuda.is_available():
                raise RuntimeError("cuda is unavailable in this worker environment")
            torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self._dtype]
            model_path = snapshot_download(
                repo_id=MODEL_ID, revision=REVISION,
                local_files_only=local_files_only,
            )
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch_dtype
            )
            model.set_processor(model_path)
            self._model = model.to(device="cuda", dtype=torch_dtype).eval()
        return self._model

    def _encode(self, **kwargs) -> np.ndarray:
        import torch

        with torch.inference_mode():
            values = self._load().encode(**kwargs)
        return values.detach().float().cpu().numpy()

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        return self._encode(images=[str(path) for path in paths])

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model = self._load()
        max_length = int(getattr(model.config.text_config, "max_position_embeddings", 77))
        inputs = model.processor(
            text=list(texts), return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model.device)
        with torch.inference_mode():
            values = model.get_text_features(**inputs)
            values = torch.nn.functional.normalize(values, dim=-1)
        return values.detach().float().cpu().numpy()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("request_path", type=Path)
    return run_worker(Adapter(), parser.parse_args(argv).request_path)


if __name__ == "__main__":
    raise SystemExit(main())
