"""Package the project for Kaggle Datasets (ADR-016).

Two bundles:

* **code**: every project file except environments, data, caches and artifacts, with a
  deterministic ``AIAYN_MANIFEST.json`` (per-file SHA-256 + a tree hash). Without git,
  the tree hash is the code version recorded by validation and training runs.
* **data**: the pinned raw archives of a translation corpus (checksum-verified again),
  the trained tokenizer and its content fingerprint.

Each bundle directory gets a ``dataset-metadata.json`` for ``kaggle datasets create``.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiayn import __version__
from aiayn.data.config import DataConfig
from aiayn.data.prepare import data_paths
from aiayn.data.tokenizer import SentencePieceTokenizer
from aiayn.utils.hashing import sha256_bytes, sha256_file

__all__ = [
    "CODE_EXCLUDES",
    "MANIFEST_NAME",
    "ROOT_EXCLUDES",
    "build_code_bundle",
    "build_data_bundle",
    "code_files",
    "tree_hash",
]

MANIFEST_NAME = "AIAYN_MANIFEST.json"

#: Excluded only as a *top-level* directory of the project. ("data" must not match the
#: aiayn.data package or configs/data/; a Week-5 dry run caught exactly that bug.)
ROOT_EXCLUDES: tuple[str, ...] = (".venv", "venv", ".git", "data", "dist", "runs", "wandb")

#: Excluded wherever they appear (glob patterns on any path component).
CODE_EXCLUDES: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".vscode",
    ".idea",
    "*.egg-info",
    ".coverage",
    ".coverage.*",
    "*.pyc",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    ".env",
    ".env.*",
    MANIFEST_NAME,
)


@dataclass(frozen=True)
class Bundle:
    directory: Path
    files: dict[str, str]  # relative path -> sha256
    tree_sha256: str


def _excluded(relative: Path, excludes: tuple[str, ...]) -> bool:
    parts = relative.parts
    if parts and parts[0] in ROOT_EXCLUDES:
        return True
    return any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in excludes)


def code_files(root: Path, excludes: tuple[str, ...] = CODE_EXCLUDES) -> list[Path]:
    """Project files to ship, as sorted POSIX-relative paths."""
    root = root.resolve()
    files: list[Path] = []
    for directory, subdirs, names in os.walk(root):
        rel_dir = Path(directory).relative_to(root)
        # Prune excluded directories in place so os.walk never descends into them (.venv!).
        subdirs[:] = [d for d in subdirs if not _excluded(rel_dir / d, excludes)]
        files.extend(rel_dir / name for name in names if not _excluded(rel_dir / name, excludes))
    return sorted(files, key=lambda p: p.as_posix())


def tree_hash(files: dict[str, str]) -> str:
    """Order-independent content hash of a ``{path: sha256}`` mapping."""
    lines = "\n".join(f"{path}\t{digest}" for path, digest in sorted(files.items()))
    return sha256_bytes(lines.encode("utf-8"))


def _metadata(slug: str, title: str, kaggle_user: str) -> dict[str, Any]:
    return {
        "title": title,
        "id": f"{kaggle_user}/{slug}",
        "licenses": [{"name": "unknown"}],
    }


def build_code_bundle(root: Path, out_dir: Path, *, kaggle_user: str) -> Bundle:
    """Copy the project to ``out_dir/aiayn/`` and write the manifest + Kaggle metadata."""
    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(f"{root} is not the project root (no pyproject.toml)")
    target = out_dir / "aiayn"
    if target.exists():
        shutil.rmtree(target)
    files: dict[str, str] = {}
    for rel in code_files(root):
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, destination)
        files[rel.as_posix()] = sha256_file(destination)
    digest = tree_hash(files)
    manifest = {"package_version": __version__, "tree_sha256": digest, "files": files}
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(_metadata("aiayn-code", "aiayn code", kaggle_user), indent=2), encoding="utf-8"
    )
    return Bundle(out_dir, files, digest)


def build_data_bundle(config: DataConfig, out_dir: Path, *, kaggle_user: str) -> Bundle:
    """Copy the raw archives + tokenizer of a translation config into ``out_dir``."""
    if config.translation is None:
        raise ValueError(f"{config.name}: only translation datasets need a data bundle")
    paths = data_paths(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for spec in config.translation.downloads:
        name = Path(urllib.parse.urlparse(spec.url).path).name
        source = paths.raw_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"{source} missing; run scripts/prepare_data.py first")
        digest = sha256_file(source)
        if digest != spec.sha256:
            raise ValueError(f"{source}: checksum {digest} != pinned {spec.sha256}")
        shutil.copyfile(source, out_dir / name)
        files[name] = digest
    model = paths.tokenizer_model
    if not model.is_file():
        raise FileNotFoundError(f"{model} missing; run scripts/train_tokenizer.py first")
    shutil.copyfile(model, out_dir / model.name)
    files[model.name] = sha256_file(model)
    fingerprint = SentencePieceTokenizer(model).fingerprint
    (out_dir / "tokenizer_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")
    (out_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())), encoding="utf-8"
    )
    slug = f"aiayn-{config.name.replace('_', '-')}"
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(_metadata(slug, f"aiayn {config.name} data", kaggle_user), indent=2),
        encoding="utf-8",
    )
    return Bundle(out_dir, files, tree_hash(files))
