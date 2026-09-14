"""Prepare / verify / restore only the runtime models used by this project.

prepare reads installed HF snapshots offline. It does not upload or download.
restore recreates the ORIGINAL pinned Hub cache paths, so workers need no model-ID edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parents[1]
HUB_MODELS = {
    "facebook/metaclip-2-worldwide-huge-quickgelu": "c139061af7b10fdb2e754b60d2b1182a3d5526c2",
    "BAAI/BGE-VL-large": "40fb48217f521df22a2a5bf15edd52ed1146ef05",
    "facebook/PE-Core-B16-224": "a16450b46fef32363459920c2685a1b4ef13dcd9",
}
LOCAL_FILES = [
    "tv2/WP03/model-cache/beit3/beit3_base_patch16_384_coco_retrieval.pth",
    "tv2/WP03/model-cache/beit3/beit3.spm",
    "tv3/WP04/weights/easyocr/craft_mlt_25k.pth",
    "tv3/WP04/weights/parseq/best-parseq.ckpt",
    "tv3/WP04/weights/rf-detr/rf-detr-medium.pth",
] + ["tv3/WP04/weights/chunkformer/" + name for name in
     ["config.yaml", "global_cmvn", "pytorch_model.bin", "vocab.txt", "README.md"]]
MODEL_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".model", ".json", ".txt", ".py", ".md"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_join(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in {"..", "."} or ":" in p or "\\" in p for p in parts.parts):
        raise ValueError(f"Unsafe bundle path: {relative}")
    target = root.joinpath(*parts.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes destination: {relative}")
    return target


def inventory() -> list[tuple[Path, str]]:
    from huggingface_hub import snapshot_download
    result = []
    for relative in LOCAL_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required runtime model missing: {relative}")
        result.append((path, "project/" + relative))
    for repo, revision in HUB_MODELS.items():
        snapshot = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        prefix = "hub/models--" + repo.replace("/", "--") + "/snapshots/" + revision
        for file in sorted(snapshot.rglob("*")):
            relative = file.relative_to(snapshot)
            if not file.is_file() or len(relative.parts) != 1 or file.name.startswith("."):
                continue
            if file.suffix not in MODEL_SUFFIXES and file.name not in {"LICENSE", "NOTICE"}:
                continue
            result.append((file, prefix + "/" + relative.as_posix()))
    return result


def prepare(bundle: Path, list_only: bool) -> None:
    files = inventory()
    for source, relative in files:
        print(f"{source.stat().st_size:>12,}  {relative}")
    print(f"Total: {len(files)} files; {sum(p.stat().st_size for p, _ in files):,} bytes")
    if list_only:
        return
    if bundle.exists():
        raise ValueError("Bundle already exists; choose a new --bundle directory.")
    if bundle.parent != ROOT or not bundle.name.startswith("_hf_export"):
        raise ValueError("Prepare output must be _hf_export* directly inside the project.")
    records = []
    for source, relative in files:
        target = safe_join(bundle, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({"path": relative, "bytes": target.stat().st_size, "sha256": digest(target)})
    shutil.copy2(ROOT / "docs/MODEL_CARD.md", bundle / "README.md")
    records.append({"path": "README.md", "bytes": (bundle / "README.md").stat().st_size,
                    "sha256": digest(bundle / "README.md")})
    (bundle / "bundle-manifest.json").write_text(json.dumps({"schema_version": 1, "hub_models": HUB_MODELS,
        "files": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared: {bundle}; run verify before uploading.")


def verify(bundle: Path) -> list[dict]:
    manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("hub_models") != HUB_MODELS:
        raise ValueError("Unexpected bundle schema or model revisions")
    records = manifest["files"]
    expected = {"bundle-manifest.json"}
    for record in records:
        relative = record["path"]
        if relative in expected:
            raise ValueError(f"Duplicate bundle entry: {relative}")
        path = safe_join(bundle, relative)
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
        expected.add(relative)
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()
              and ".cache" not in p.relative_to(bundle).parts}
    if actual != expected:
        raise ValueError(f"Unexpected/missing files: {sorted(actual ^ expected)}")
    required = {"project/" + p for p in LOCAL_FILES}
    if not required <= expected:
        raise ValueError("Required runtime weights missing from manifest")
    print(f"Verified {len(records)} files.")
    return records


def restore(bundle: Path, hub_cache: Path) -> None:
    records = verify(bundle)
    targets = []
    for record in records:
        relative = record["path"]
        if relative == "README.md":
            continue
        if relative.startswith("project/") and relative[8:] in LOCAL_FILES:
            target = safe_join(ROOT, relative[8:])
        elif relative.startswith("hub/") and any(relative.startswith("hub/models--" + repo.replace("/", "--") +
              "/snapshots/" + revision + "/") for repo, revision in HUB_MODELS.items()):
            target = safe_join(hub_cache, relative[4:])
        else:
            raise ValueError(f"Unapproved restore destination: {relative}")
        if target.exists() and digest(target) != record["sha256"]:
            raise ValueError(f"Existing file differs; no overwrite: {target}")
        targets.append((safe_join(bundle, relative), target))
    for source, target in targets:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"Restored model files. Set HF_HUB_CACHE={hub_cache} in EVERY TV2/TV4 terminal.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "restore"])
    parser.add_argument("--bundle", type=Path, default=ROOT / "_hf_export")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--hub-cache", type=Path, default=ROOT / ".cache/huggingface/hub")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.command == "prepare":
        prepare(bundle, args.list)
    elif args.command == "verify":
        verify(bundle)
    else:
        restore(bundle, args.hub_cache.resolve())


if __name__ == "__main__":
    main()
