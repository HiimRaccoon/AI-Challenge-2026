from __future__ import annotations

import json
from pathlib import Path

import pytest

from wp03.contracts import ContractError
from wp03.config import RuntimeProfile
from wp03.worker_protocol import WorkerRequest, WorkerStatus, compatibility_fingerprint, runtime_fingerprint


def test_status_with_another_job_id_is_rejected(tmp_path: Path) -> None:
    request = WorkerRequest.create(
        job_dir=tmp_path,
        operation="encode_text",
        model_key="beit3",
        revision="rev",
        device="cpu",
        dtype="float32",
        batch_size=1,
        attempt=1,
        image_paths=(),
        texts=("car",),
    )
    request.write()
    request.status_path.write_text(
        json.dumps({"job_id": "other-job", "request_sha256": request.sha256, "status": "failed"}),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="job_id"):
        WorkerStatus.from_json(request.status_path, request)


def test_success_status_without_embedding_metadata_is_rejected(tmp_path: Path) -> None:
    request = WorkerRequest.create(
        job_dir=tmp_path,
        operation="encode_text",
        model_key="beit3",
        revision="rev",
        device="cpu",
        dtype="float32",
        batch_size=1,
        attempt=1,
        image_paths=(),
        texts=("car",),
    )
    request.write()
    request.status_path.write_text(
        json.dumps({"job_id": request.job_id, "request_sha256": request.sha256, "status": "ok"}),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="count"):
        WorkerStatus.from_json(request.status_path, request)


@pytest.mark.parametrize(("batch_size", "attempt"), [(0, 1), (1, 0)])
def test_worker_request_rejects_nonpositive_batch_or_attempt(tmp_path: Path, batch_size: int, attempt: int) -> None:
    with pytest.raises(ContractError):
        WorkerRequest.create(tmp_path, "encode_images", "beit3", "rev", "cpu", "float32", batch_size, attempt, (), ())


def test_runtime_fingerprint_can_change_without_changing_compatibility() -> None:
    compatibility = compatibility_fingerprint(revision="a", tokenizer_revision="one")
    assert runtime_fingerprint(compatibility=compatibility, dtype="bfloat16") != runtime_fingerprint(
        compatibility=compatibility, dtype="float16"
    )


def test_windows_runtime_profile_resolves_worker_under_runtime_root(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text(
        "workers:\n  bge_vl: ['.venvs/bge_vl/Scripts/python.exe', '-m', 'worker']\n",
        encoding="utf-8",
    )

    profile = RuntimeProfile.load(profile_path, tmp_path)

    assert profile.command_for("bge_vl")[0] == str(tmp_path / ".venvs/bge_vl/Scripts/python.exe")


def test_runtime_profile_resolves_relative_worker_environment(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text(
        "workers:\n  beit3:\n    command: ['.venvs/beit3/Scripts/python.exe', '-m', 'worker']\n    env:\n      WP03_BEIT3_CHECKPOINT: 'model-cache/beit3/model.pth'\n",
        encoding="utf-8",
    )

    profile = RuntimeProfile.load(profile_path, tmp_path)

    assert profile.environment_for("beit3")["WP03_BEIT3_CHECKPOINT"] == str(tmp_path / "model-cache/beit3/model.pth")
