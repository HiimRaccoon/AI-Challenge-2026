from __future__ import annotations

import json
import sys

import numpy as np

from wp03.worker_launcher import WorkerProcessEncoder


def test_subprocess_launcher_accepts_checked_numpy_output(tmp_path) -> None:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """
import hashlib, json, sys
from pathlib import Path
import numpy as np
request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
output = Path(request['output_path'])
with output.open('wb') as stream:
    np.save(stream, np.array([[1., 0.]], dtype=np.float32), allow_pickle=False)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
Path(request['status_path']).write_text(json.dumps({
    'job_id': request['job_id'], 'request_sha256': request['request_sha256'],
    'status': 'ok', 'output_sha256': digest, 'count': 1, 'dimension': 2,
    'dtype': 'float32', 'normalized': True
}), encoding='utf-8')
""",
        encoding="utf-8",
    )
    encoder = WorkerProcessEncoder(
        command=(sys.executable, str(worker)), job_root=tmp_path / "jobs", model_key="fake",
        revision="rev", device="cpu", dtype="float32", batch_size=1,
    )

    vectors = encoder.encode_text(["car"])

    assert np.array_equal(vectors, np.array([[1.0, 0.0]], dtype=np.float32))


def test_retryable_oom_restarts_worker_once_with_half_batch(tmp_path) -> None:
    worker = tmp_path / "oom_then_success.py"
    worker.write_text(
        """
import hashlib, json, sys
from pathlib import Path
import numpy as np
request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if request['attempt'] == 1:
    Path(request['status_path']).write_text(json.dumps({
        'job_id': request['job_id'], 'request_sha256': request['request_sha256'],
        'status': 'failed', 'error_type': 'cuda_oom', 'message': 'out of memory', 'retryable': True
    }), encoding='utf-8')
    raise SystemExit(1)
output = Path(request['output_path'])
with output.open('wb') as stream:
    np.save(stream, np.array([[1., 0.]], dtype=np.float32), allow_pickle=False)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
Path(request['status_path']).write_text(json.dumps({
    'job_id': request['job_id'], 'request_sha256': request['request_sha256'], 'status': 'ok',
    'output_sha256': digest, 'count': 1, 'dimension': 2, 'dtype': 'float32', 'normalized': True
}), encoding='utf-8')
""",
        encoding="utf-8",
    )
    encoder = WorkerProcessEncoder(
        command=(sys.executable, str(worker)), job_root=tmp_path / "jobs", model_key="fake",
        revision="rev", device="cpu", dtype="float32", batch_size=4,
    )

    result = encoder.encode_text(["car"])

    requests = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "jobs").glob("*.request.json")),
        key=lambda request: request["attempt"],
    )
    assert np.array_equal(result, np.array([[1.0, 0.0]], dtype=np.float32))
    assert [(request["attempt"], request["batch_size"]) for request in requests] == [(1, 4), (2, 2)]


def test_retryable_oom_keeps_halving_batch_until_success(tmp_path) -> None:
    worker = tmp_path / "oom_until_batch_one.py"
    worker.write_text(
        """
import hashlib, json, sys
from pathlib import Path
import numpy as np
request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if request['batch_size'] > 1:
    Path(request['status_path']).write_text(json.dumps({
        'job_id': request['job_id'], 'request_sha256': request['request_sha256'],
        'status': 'failed', 'error_type': 'cuda_oom', 'message': 'out of memory', 'retryable': True
    }), encoding='utf-8')
    raise SystemExit(1)
output = Path(request['output_path'])
with output.open('wb') as stream:
    np.save(stream, np.array([[1., 0.]], dtype=np.float32), allow_pickle=False)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
Path(request['status_path']).write_text(json.dumps({
    'job_id': request['job_id'], 'request_sha256': request['request_sha256'], 'status': 'ok',
    'output_sha256': digest, 'count': 1, 'dimension': 2, 'dtype': 'float32', 'normalized': True
}), encoding='utf-8')
""",
        encoding="utf-8",
    )
    encoder = WorkerProcessEncoder(
        command=(sys.executable, str(worker)), job_root=tmp_path / "jobs", model_key="fake",
        revision="rev", device="cpu", dtype="float32", batch_size=16,
    )

    result = encoder.encode_text(["car"])

    requests = sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "jobs").glob("*.request.json")),
        key=lambda request: request["attempt"],
    )
    assert np.array_equal(result, np.array([[1.0, 0.0]], dtype=np.float32))
    assert [request["batch_size"] for request in requests] == [16, 8, 4, 2, 1]
