from __future__ import annotations

import pytest

from wp03.contracts import ContractError
from wp03.model_lock import ModelIdentity, create_model_lock, validate_model_lock
from wp03.workers.beit3 import BEIT3_EXPECTED_SIZE_BYTES, BEIT3_OFFICIAL_URL


def test_model_lock_rejects_digest_change_at_same_size(tmp_path) -> None:
    checkpoint = tmp_path / "beit3.pth"
    checkpoint.write_bytes(b"trusted-test-bytes")
    identity = ModelIdentity(name="beit3", source_url=BEIT3_OFFICIAL_URL)
    lock = create_model_lock(checkpoint, tmp_path / "beit3.json", identity, len(b"trusted-test-bytes"))

    checkpoint.write_bytes(b"altered-test-bytes")

    with pytest.raises(ContractError, match="digest"):
        validate_model_lock(lock.path, checkpoint)


def test_beit3_lock_refuses_unofficial_size(tmp_path) -> None:
    checkpoint = tmp_path / "beit3.pth"
    checkpoint.write_bytes(b"small")

    with pytest.raises(ContractError, match="size"):
        create_model_lock(
            checkpoint,
            tmp_path / "beit3.json",
            ModelIdentity(name="beit3", source_url=BEIT3_OFFICIAL_URL),
            BEIT3_EXPECTED_SIZE_BYTES,
        )
