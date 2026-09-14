"""Lazy Hugging Face dual-encoder helper used only inside a worker process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np


class HuggingFaceClipAdapter:
    model_id: str
    revision: str

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = "float32"

    def configure(self, *, dtype: str) -> None:
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported worker dtype")
        self._dtype = dtype

    def _load(self) -> None:
        if self._model is not None:
            return
        local_files_only = os.environ.get("WP03_HF_LOCAL_ONLY", "1") != "0"
        if local_files_only:
            # ``local_files_only`` alone can still trigger Hub metadata calls
            # in some Transformers/Hugging Face Hub version combinations.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModel, AutoProcessor

        requested = os.environ.get("WP03_DEVICE", "cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda is unavailable in this worker environment")
        self._device = torch.device(requested)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self._dtype]
        # Online retries turn an unavailable optional model into a multi-minute
        # query stall.  WP03 must use the reviewed local model cache by default;
        # an operator can explicitly opt into downloading during provisioning.
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        ).to(self._device).eval()

    def _as_numpy(self, value) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        self._load()
        from PIL import Image
        import torch

        images = [Image.open(path).convert("RGB") for path in paths]
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {
            name: value.to(self._device, dtype=self._model.dtype) if value.is_floating_point() else value.to(self._device)
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            values = self._model.get_image_features(**inputs)
        return self._as_numpy(values)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        import torch

        inputs = self._processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs = {
            name: value.to(self._device, dtype=self._model.dtype) if value.is_floating_point() else value.to(self._device)
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            values = self._model.get_text_features(**inputs)
        return self._as_numpy(values)
