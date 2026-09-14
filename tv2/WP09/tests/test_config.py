from __future__ import annotations

from pathlib import Path

from wp09.config import RefinementConfig


def test_default_config_is_a_safe_single_model_refinement_profile() -> None:
    """Catches a default that unexpectedly batches models or removes manual fallback."""

    config = RefinementConfig.load(Path("configs/default.yaml"))

    assert config.refiner_model == "google/siglip2-base-patch16-224"
    assert config.batch_size == 8
    assert config.cache_max_entries == 32
    assert config.cache_ttl_seconds == 300.0
    assert config.failure_mode == "manual_only"
