"""Boundary checks for packages that will be published externally."""
import importlib.util
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


export = module('export_github')
models = module('model_bundle')


def test_export_excludes_private_and_generated_files():
    paths = [p.relative_to(export.ROOT).as_posix() for p in export.selected_files()]
    assert 'tv5/tsconfig.build.json' in paths
    assert 'tv2/WP03/envs/metaclip2.txt' in paths
    assert 'scripts/model_bundle.py' in paths
    for path in paths:
        assert not any(part in {'.git', 'data', 'weights', 'node_modules', 'outputs', '__pycache__'}
                       for part in Path(path).parts)
        assert not path.startswith(('submission', 'DATASETS_ZIP', 'weights', 'evaluation', '_hf_export'))
        assert Path(path).suffix not in {'.zip', '.parquet', '.faiss', '.pth', '.ckpt', '.safetensors'}


@pytest.mark.parametrize('path', ['../outside', '/absolute', 'C:/outside', 'hub/../../outside', 'hub\\..\\outside'])
def test_restore_rejects_escaping_paths(tmp_path, path):
    with pytest.raises(ValueError):
        models.safe_join(tmp_path, path)


def test_restore_accepts_nested_runtime_path(tmp_path):
    assert models.safe_join(tmp_path, 'hub/model/config.json') == tmp_path / 'hub/model/config.json'
