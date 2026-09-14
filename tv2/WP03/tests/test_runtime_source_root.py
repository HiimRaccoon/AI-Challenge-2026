from pathlib import Path

from wp03.config import RuntimeProfile


def test_windows_runtime_uses_current_source_for_every_worker():
    root = Path(__file__).resolve().parents[1]
    profile = RuntimeProfile.load(root / "configs" / "runtime.windows.yaml", root)
    for model_key in ("beit3", "bge_vl", "metaclip2", "perception"):
        assert Path(profile.environment_for(model_key)["WP03_SOURCE_ROOT"]) == (root / "src").resolve()
