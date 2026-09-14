"""Export reviewed source paths without inheriting the working repository's history.

No git mutation, network access, deletion or overwrite. Run from any directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = [".gitignore", "README.md", "tv1/requirements.txt", "tv1/config_example.yaml"]
TREES = ["docs", "scripts", "tv1/tests"]
for module in ["tv2/WP03", "tv2/WP08", "tv2/WP09", "tv3/WP04", "tv4"]:
    FILES += [f"{module}/pyproject.toml"]
    TREES += [f"{module}/src", f"{module}/tests", f"{module}/configs"]
FILES += ["tv4/requirements.txt", "tv5/package.json", "tv5/package-lock.json",
          "tv5/index.html", "tv5/tsconfig.json", "tv5/tsconfig.node.json",
          "tv5/vite.config.ts", "tv5/vitest.config.ts", "tv5/tsconfig.build.json"]
TREES += ["tv2/WP03/envs", "tv3/WP04/deploy/model-server", "tv5/src", "tv5/tests",
          "tv5/backup", "tv5/deployment", "tv5/evaluation", "tv5/governance",
          "tv5/readiness", "tv5/submission", "tv5/telemetry"]
FILES += ["tv5/__init__.py"]
SUFFIXES = {".py", ".ts", ".tsx", ".css", ".html", ".json", ".yaml", ".yml",
            ".toml", ".md", ".txt", ".ps1", ".example"}
EXCLUDE = {"tv3/WP04/deploy/model-server/inspect_weights_zip.py",
           "tv3/WP04/deploy/model-server/README.md"}


def selected_files(root: Path = ROOT) -> list[Path]:
    candidates = [root / name for name in FILES]
    candidates += list((root / "tv1").glob("wp*.py"))
    for name in TREES:
        candidates += list((root / name).rglob("*"))
    selected = []
    for path in sorted(set(candidates)):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Path escapes source root: {relative}")
        if any(p.startswith('.') or p in {"__pycache__", "node_modules", "outputs", "data"}
               for p in relative.parts[:-1]):
            continue
        if relative.as_posix() in EXCLUDE or "certifications" in relative.parts:
            continue
        if path.name.startswith(".env") or path.name.endswith(".local.yaml"):
            continue
        if path.suffix not in SUFFIXES and relative.as_posix() != ".gitignore":
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Unexpected large source file: {relative}")
        selected.append(path)
    for name in FILES:
        if root / name not in selected:
            raise ValueError(f"Required source missing/excluded: {name}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_github_export")
    parser.add_argument("--list", action="store_true", help="Print selected paths without copying")
    args = parser.parse_args()
    paths = selected_files()
    if args.list:
        for path in paths:
            print(path.relative_to(ROOT).as_posix())
        return
    destination = args.output.resolve()
    if destination.exists():
        raise SystemExit("Output already exists. Choose a NEW --output directory.")
    if not destination.is_relative_to(ROOT) or destination.parent != ROOT or not destination.name.startswith("_github_export"):
        raise SystemExit("Output must be a new _github_export* directory directly inside the project.")
    records = []
    for source in paths:
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with target.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                        "sha256": checksum})
    # Store inventory outside the tree that will be pushed.
    report = ROOT / (destination.name + "_manifest.json")
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(records)} files, {sum(r['bytes'] for r in records):,} bytes to {destination}")
    print(f"Review inventory: {report}")


if __name__ == "__main__":
    main()
