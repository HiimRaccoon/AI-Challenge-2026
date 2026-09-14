from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from wp03.contracts import ContractError
from wp03.worker_protocol import WorkerRequest
from wp03.workers.beit3 import Adapter as Beit3Adapter
from wp03.workers.common import run_worker
from wp03.workers.perception import Adapter as PerceptionAdapter


class FakeAdapter:
    def encode_images(self, paths):
        return np.ones((len(paths), 2), dtype=np.float32)

    def encode_text(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)


def test_worker_writes_failed_status_for_unknown_operation(tmp_path) -> None:
    request = WorkerRequest.create(
        tmp_path, "encode_images", "fake", "rev", "cpu", "float32", 1, 1, (), ()
    )
    payload = request.to_dict()
    payload["operation"] = "bad"
    request.request_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = run_worker(FakeAdapter(), request.request_path)

    status = json.loads(request.status_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert status["error_type"] == "unsupported_operation"


def test_beit3_adapter_refuses_missing_model_lock_before_backend_import(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="model lock"):
        Beit3Adapter.from_runtime(
            checkpoint_path=tmp_path / "beit3.pth",
            lock_path=tmp_path / "beit3.json",
            source_dir=tmp_path / "unilm" / "beit3",
        )


def test_perception_load_uses_registry_name_with_pinned_hub_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: PE's registry accepts its short config name, not its Hub ID."""

    calls: dict[str, object] = {}

    class FakeModel:
        image_size = 224
        context_length = 32

        def to(self, *, device: str, dtype: object) -> "FakeModel":
            calls["device"] = device
            calls["dtype"] = dtype
            return self

        def eval(self) -> "FakeModel":
            return self

    class FakeClip:
        @classmethod
        def from_config(cls, name: str, *, pretrained: bool, checkpoint_path: str) -> FakeModel:
            if name != "PE-Core-B16-224":
                raise RuntimeError(f"{name} not found in configs.")
            calls["config_name"] = name
            calls["pretrained"] = pretrained
            calls["checkpoint_path"] = checkpoint_path
            return FakeModel()

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_core = ModuleType("core")
    fake_core.__path__ = []
    fake_vision_encoder = ModuleType("core.vision_encoder")
    fake_vision_encoder.__path__ = []
    fake_pe = ModuleType("core.vision_encoder.pe")
    fake_pe.CLIP = FakeClip
    fake_transforms = ModuleType("core.vision_encoder.transforms")
    fake_transforms.get_image_transform = lambda image_size: image_size
    fake_transforms.get_text_tokenizer = lambda context_length: context_length
    fake_hub = ModuleType("huggingface_hub")

    def fake_hub_download(*, repo_id: str, filename: str, revision: str) -> str:
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        calls["revision"] = revision
        return "C:/cache/PE-Core-B16-224.pt"

    fake_hub.hf_hub_download = fake_hub_download
    for name, module in {
        "torch": fake_torch,
        "core": fake_core,
        "core.vision_encoder": fake_vision_encoder,
        "core.vision_encoder.pe": fake_pe,
        "core.vision_encoder.transforms": fake_transforms,
        "huggingface_hub": fake_hub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    adapter = PerceptionAdapter()
    adapter._load()

    assert calls == {
        "repo_id": "facebook/PE-Core-B16-224",
        "filename": "PE-Core-B16-224.pt",
        "revision": "a16450b46fef32363459920c2685a1b4ef13dcd9",
        "config_name": "PE-Core-B16-224",
        "pretrained": True,
        "checkpoint_path": "C:/cache/PE-Core-B16-224.pt",
        "device": "cuda",
        "dtype": "bfloat16",
    }


def test_worker_applies_request_batch_size_without_reordering(tmp_path: Path) -> None:
    class BatchRecordingAdapter(FakeAdapter):
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def encode_text(self, texts):
            self.batch_sizes.append(len(texts))
            return np.asarray([[float(index + 1), 0.0] for index in range(len(texts))], dtype=np.float32)

    adapter = BatchRecordingAdapter()
    request = WorkerRequest.create(
        tmp_path, "encode_text", "fake", "rev", "cpu", "float32", 2, 1, (), ("one", "two", "three", "four", "five")
    )
    request.write()

    assert run_worker(adapter, request.request_path) == 0
    assert adapter.batch_sizes == [2, 2, 1]
