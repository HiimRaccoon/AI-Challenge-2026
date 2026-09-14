"""Dependency-free verification of the derived WP13 traceability map."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

FR_RE = re.compile(r"^- \*\*(FR-\d{3})\*\*:", re.M)
TASK_RE = re.compile(r"^- \[[ x]\] \*\*(T\d{3})\b(?P<body>.*)$", re.M)
FR_RANGE_RE = re.compile(r"FR-(\d{3})(?:-(\d{3}))?")
FORBIDDEN = re.compile(r"\b(?:run|rerun|start|schedule|implement|create|build|regenerate|repair)\b[^.]*\b(?:WP03|WP04)?\s*(?:preprocess(?:ing)?|rebuild|persistent .*frame[- ]map|replacement frame mapping)", re.I)

@dataclass(frozen=True)
class TaskRecord:
    id: str
    requirements: tuple[str, ...]
    prerequisites: tuple[str, ...]
    body: str

@dataclass(frozen=True)
class TraceabilityReport:
    ok: bool
    diagnostics: tuple[str, ...]

def _duplicates(items: list[str]) -> list[str]:
    return sorted({item for item in items if items.count(item) > 1})

def _expand_ranges(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in FR_RANGE_RE.finditer(text):
        start, end = int(match.group(1)), int(match.group(2) or match.group(1))
        values.extend(f"FR-{number:03d}" for number in range(start, end + 1))
    return tuple(values)

def parse_authoritative_tasks(text: str) -> tuple[TaskRecord, ...]:
    """Parse the stable one-line task bullet format in the authoritative file."""
    records: list[TaskRecord] = []
    for match in TASK_RE.finditer(text):
        body = match.group("body")
        req_match = re.search(r"Reqs:\s*([^)]*)", body)
        prereq_match = re.search(r"Prereq:\s*([^.]*)", body)
        records.append(TaskRecord(
            id=match.group(1),
            requirements=_expand_ranges(req_match.group(1) if req_match else ""),
            prerequisites=tuple(re.findall(r"T\d{3}", prereq_match.group(1) if prereq_match else "")),
            body=body,
        ))
    return tuple(records)

def check_traceability(path: Path) -> TraceabilityReport:
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent
        spec = (root / data["sources"]["spec"]).read_text(encoding="utf-8")
        task_text = (root / data["sources"]["tasks"]).read_text(encoding="utf-8")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return TraceabilityReport(False, (f"unreadable traceability input: {exc}",))
    diagnostics: list[str] = []
    frs, authoritative_tasks = FR_RE.findall(spec), parse_authoritative_tasks(task_text)
    task_ids = [task.id for task in authoritative_tasks]
    expected_frs, expected_tasks = list(data.get("expected_fr_ids", [])), list(data.get("expected_task_ids", []))
    for label, actual, expected in (("FR", frs, expected_frs), ("Task", task_ids, expected_tasks)):
        if duplicate := _duplicates(actual): diagnostics.append(f"duplicate {label} definitions: {', '.join(duplicate)}")
        if missing := sorted(set(expected) - set(actual)): diagnostics.append(f"missing {label} definitions: {', '.join(missing)}")
        if extra := sorted(set(actual) - set(expected)): diagnostics.append(f"unexpected {label} definitions: {', '.join(extra)}")
    authoritative = {task.id: task for task in authoritative_tasks}
    for task in authoritative_tasks:
        if unknown := sorted(set(task.requirements) - set(frs)): diagnostics.append(f"task {task.id} references nonexistent FR: {', '.join(unknown)}")
        if unknown := sorted(set(task.prerequisites) - set(task_ids)): diagnostics.append(f"task {task.id} has nonexistent prerequisite: {', '.join(unknown)}")
        if FORBIDDEN.search(task.body) and not re.search(r"\b(?:no|never|do not)\s+(?:WP13\s+)?(?:repair|preprocess|rebuild)", task.body, re.I): diagnostics.append(f"forbidden WP13 preprocessing/rebuild in authoritative task: {task.id}")
    mapped_tasks = data.get("tasks", [])
    mapped_ids = [task.get("id", "") for task in mapped_tasks]
    if duplicate := _duplicates(mapped_ids): diagnostics.append(f"duplicate Task IDs in traceability map: {', '.join(duplicate)}")
    if set(mapped_ids) != set(task_ids): diagnostics.append("traceability map has missing or extra Task IDs")
    for item in mapped_tasks:
        task_id = item.get("id", "")
        if task_id not in authoritative: continue
        actual = authoritative[task_id]
        if tuple(item.get("requirements", [])) != actual.requirements:
            diagnostics.append(f"traceability requirements disagree with tasks.md: {task_id}")
        if tuple(item.get("prerequisites", [])) != actual.prerequisites:
            diagnostics.append(f"traceability prerequisites disagree with tasks.md: {task_id}")
    mapped_frs = data.get("requirements", [])
    ids = [item.get("id", "") for item in mapped_frs]
    if duplicate := _duplicates(ids): diagnostics.append(f"duplicate FR map entries: {', '.join(duplicate)}")
    if set(ids) != set(frs): diagnostics.append("orphan or missing FR map entries")
    for item in mapped_frs:
        fr = item.get("id", "")
        linked = tuple(item.get("tasks", []))
        if not item.get("tests"): diagnostics.append(f"orphan FR without planned/implemented test evidence: {fr}")
        if unknown := sorted(set(linked) - set(task_ids)): diagnostics.append(f"FR {fr} maps to nonexistent task: {', '.join(unknown)}")
        authoritative_links = tuple(task.id for task in authoritative_tasks if fr in task.requirements)
        if linked != authoritative_links:
            diagnostics.append(f"traceability FR-task mapping disagrees with tasks.md: {fr}")
    return TraceabilityReport(not diagnostics, tuple(diagnostics))
