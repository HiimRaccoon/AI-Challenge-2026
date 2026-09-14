"""Pinned BEiT-3 COCO-retrieval worker backed by the official UniLM checkout."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..contracts import ContractError
from ..model_lock import ModelIdentity, validate_model_lock
from .common import run_worker

REVISION = "833df7e7832e5064a281131ee64a481afa8e5b95"
BEIT3_EXPECTED_SIZE_BYTES = 445_025_515
BEIT3_OFFICIAL_URL = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth"
MODEL_NAME = "beit3_base_patch16_384_retrieval"
IMAGE_SIZE = 384
MAX_TEXT_TOKENS = 64


@dataclass
class Adapter:
    checkpoint_path: Path | None = None
    lock_path: Path | None = None
    source_dir: Path | None = None
    sentencepiece_path: Path | None = None
    dtype: str = "bfloat16"
    _model: object | None = None
    _tokenizer: object | None = None
    _transform: object | None = None

    @classmethod
    def identity(cls) -> ModelIdentity:
        return ModelIdentity("beit3", BEIT3_OFFICIAL_URL, REVISION)

    @classmethod
    def from_runtime(
        cls, *, checkpoint_path: Path, lock_path: Path, source_dir: Path, sentencepiece_path: Path | None = None
    ) -> "Adapter":
        validate_model_lock(lock_path, checkpoint_path)
        if not source_dir.is_dir():
            raise ContractError("BEiT-3 UniLM source directory is absent")
        tokenizer_path = sentencepiece_path or source_dir / "beit3.spm"
        if not tokenizer_path.is_file():
            raise ContractError("BEiT-3 sentencepiece model is absent")
        return cls(checkpoint_path, lock_path, source_dir, tokenizer_path)

    @classmethod
    def from_environment(cls) -> "Adapter":
        def required_path(name: str) -> Path:
            value = os.environ.get(name)
            if not value:
                raise ContractError(f"{name} must identify the BEiT-3 runtime artifact")
            return Path(value)

        return cls(
            checkpoint_path=required_path("WP03_BEIT3_CHECKPOINT"),
            lock_path=required_path("WP03_BEIT3_LOCK_PATH"),
            source_dir=required_path("WP03_BEIT3_SOURCE_DIR"),
            sentencepiece_path=required_path("WP03_BEIT3_SENTENCEPIECE"),
        )

    def configure(self, *, dtype: str) -> None:
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ContractError("BEiT-3 worker dtype is unsupported")
        self.dtype = dtype

    def _load(self) -> tuple[object, object, object]:
        if self._model is not None:
            assert self._tokenizer is not None and self._transform is not None
            return self._model, self._tokenizer, self._transform
        if self.checkpoint_path is None or self.lock_path is None or self.source_dir is None or self.sentencepiece_path is None:
            raise ContractError("BEiT-3 runtime paths are absent")
        verified = self.from_runtime(
            checkpoint_path=self.checkpoint_path,
            lock_path=self.lock_path,
            source_dir=self.source_dir,
            sentencepiece_path=self.sentencepiece_path,
        )
        source_text = str(verified.source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            import torch
            import modeling_finetune  # noqa: F401  # registers the pinned retrieval architecture with timm
            import utils as beit3_utils
            from timm.models import create_model
            from torchvision import transforms
            from transformers import XLMRobertaTokenizer
        except ImportError as exc:
            raise ContractError("BEiT-3 worker dependencies are not installed") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is unavailable in this worker environment")
        model = create_model(MODEL_NAME, pretrained=False, vocab_size=64010)
        beit3_utils.load_model_and_may_interpolate(str(self.checkpoint_path), model, "model|module", "")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]
        self._model = model.to(device="cuda", dtype=torch_dtype).eval()
        self._tokenizer = XLMRobertaTokenizer(str(self.sentencepiece_path))
        self._transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        return self._model, self._tokenizer, self._transform

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        model, _, transform = self._load()
        import torch
        from PIL import Image

        pixels = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=next(model.parameters()).dtype):
            image_features, _ = model(image=pixels, only_infer=True)
        return image_features.detach().float().cpu().numpy()

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        model, tokenizer, _ = self._load()
        import torch

        encoded = tokenizer(list(texts), padding="max_length", truncation=True, max_length=MAX_TEXT_TOKENS, return_tensors="pt")
        tokens = encoded["input_ids"].to("cuda")
        padding_mask = (encoded["attention_mask"] == 0).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=next(model.parameters()).dtype):
            _, text_features = model(text_description=tokens, padding_mask=padding_mask, only_infer=True)
        return text_features.detach().float().cpu().numpy()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("request_path", type=Path)
    args = parser.parse_args(argv)
    return run_worker(Adapter.from_environment(), args.request_path)


if __name__ == "__main__":
    raise SystemExit(main())
