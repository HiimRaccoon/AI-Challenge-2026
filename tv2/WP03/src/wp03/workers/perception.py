"""Pinned Perception Encoder adapter identity and worker entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..model_lock import ModelIdentity
from .common import run_worker

REVISION = "a16450b46fef32363459920c2685a1b4ef13dcd9"
MODEL_ID = "facebook/PE-Core-B16-224"
MODEL_CONFIG_NAME = "PE-Core-B16-224"
OFFICIAL_URL = f"https://huggingface.co/{MODEL_ID}"


class Adapter:
    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dtype = "bfloat16"

    def configure(self, *, dtype: str) -> None:
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported worker dtype")
        self._dtype = dtype

    def _torch_dtype(self):
        import torch

        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self._dtype]

    @classmethod
    def identity(cls) -> ModelIdentity:
        return ModelIdentity("perception", OFFICIAL_URL, REVISION)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        from huggingface_hub import hf_hub_download

        if not torch.cuda.is_available():
            raise RuntimeError("cuda is unavailable in this worker environment")
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="PE-Core-B16-224.pt",
            revision=REVISION,
        )
        self._model = pe.CLIP.from_config(
            MODEL_CONFIG_NAME, pretrained=True, checkpoint_path=checkpoint_path
        ).to(device="cuda", dtype=self._torch_dtype()).eval()
        self._preprocess = transforms.get_image_transform(self._model.image_size)
        self._tokenizer = transforms.get_text_tokenizer(self._model.context_length)

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        self._load()
        import torch
        from PIL import Image

        pixels = torch.stack([self._preprocess(Image.open(path).convert("RGB")) for path in paths]).cuda()
        tokens = self._tokenizer([""])
        with torch.inference_mode(), torch.autocast("cuda", dtype=self._torch_dtype()):
            image_features, _, _ = self._model(pixels, tokens.cuda())
        return image_features.detach().float().cpu().numpy()

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        import torch

        tokens = self._tokenizer(list(texts)).cuda()
        pixels = torch.zeros((1, 3, self._model.image_size, self._model.image_size), device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=self._torch_dtype()):
            _, text_features, _ = self._model(pixels, tokens)
        return text_features.detach().float().cpu().numpy()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("request_path", type=Path)
    return run_worker(Adapter(), parser.parse_args(argv).request_path)


if __name__ == "__main__":
    raise SystemExit(main())
