"""RED-first acceptance tests for the WP13 governance/readiness foundation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_traceability_checker_accepts_authoritative_map_and_rejects_malformed(tmp_path: Path) -> None:
    from tv5.governance.traceability import check_traceability

    report = check_traceability(ROOT / "specs" / "001-contest-ready-wp13" / "traceability.json")
    assert report.ok, report.diagnostics
    broken = json.loads((ROOT / "specs" / "001-contest-ready-wp13" / "traceability.json").read_text())
    broken["requirements"].append(dict(broken["requirements"][0]))
    source_dir = ROOT / "specs" / "001-contest-ready-wp13"
    (tmp_path / "spec.md").write_text((source_dir / "spec.md").read_text())
    (tmp_path / "tasks.md").write_text((source_dir / "tasks.md").read_text())
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken))
    assert not check_traceability(path).ok

@pytest.mark.parametrize("mutation", ["mapping", "prereq", "bad_fr", "bad_prereq", "forbidden", "no_tests"])
def test_traceability_adversarial_authority_drift(tmp_path: Path, mutation: str) -> None:
    from tv5.governance.traceability import check_traceability
    source = ROOT / "specs" / "001-contest-ready-wp13"; data=json.loads((source/"traceability.json").read_text())
    spec=(source/"spec.md").read_text(); tasks=(source/"tasks.md").read_text()
    if mutation == "mapping": data["tasks"][1]["requirements"]=["FR-058"]
    elif mutation == "prereq": data["tasks"][3]["prerequisites"]=[]
    elif mutation == "bad_fr": tasks=tasks.replace("Reqs: FR-058-067", "Reqs: FR-999", 1)
    elif mutation == "bad_prereq": tasks=tasks.replace("Prereq: T001", "Prereq: T999", 1)
    elif mutation == "forbidden": tasks=tasks.replace("Test first: checker", "Run WP03 preprocessing and rebuild indexes. Test first: checker", 1)
    else: data["requirements"][0]["tests"]=[]
    (tmp_path/"spec.md").write_text(spec); (tmp_path/"tasks.md").write_text(tasks); (tmp_path/"map.json").write_text(json.dumps(data))
    assert not check_traceability(tmp_path/"map.json").ok


def test_readiness_statuses_and_non_mutation(tmp_path: Path) -> None:
    from tv5.readiness import ReadinessConfig, validate_readiness

    root = tmp_path / "wp03"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "preprocess_run_id": "run-a", "videos": ["v1", "v2"],
        "selected_frames": 2, "vectors": 2, "canonical_mapping": True, "model_version": "m", "index_version": "i",
    }))
    report = validate_readiness(ReadinessConfig.wp03(root, expected_videos=["v1", "v2", "v3"]))
    assert report.status == "PARTIAL"
    assert (root / "manifest.json").exists()


@pytest.mark.parametrize("kind,expected", [
    ("pending", "HANDOVER PENDING"), ("adapter", "CODE GAP"),
    ("missing", "ACTUALLY MISSING"),
])
def test_readiness_preserves_handover_and_code_gap(tmp_path: Path, kind: str, expected: str) -> None:
    from tv5.readiness import ReadinessConfig, validate_readiness

    config = ReadinessConfig(component="OCR", kind="wp04", root=tmp_path / "none", known_handover=kind == "pending", upstream_capability=kind == "adapter")
    assert validate_readiness(config).status == expected


def test_readiness_reports_incompatible_manifest(tmp_path: Path) -> None:
    from tv5.readiness import ReadinessConfig, validate_readiness

    root = tmp_path / "bad"
    root.mkdir()
    (root / "manifest.json").write_text("not-json")
    report = validate_readiness(ReadinessConfig.wp03(root, expected_videos=["v1"]))
    assert report.status == "INCOMPATIBLE"
    assert report.diagnostics
