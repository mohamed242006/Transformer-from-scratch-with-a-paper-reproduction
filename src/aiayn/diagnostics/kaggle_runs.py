"""Preserve run directories across Kaggle sessions (ADR-020).

Two layers, both driven by ``notebooks/kaggle_train.ipynb``:

1. Commit mode (*Save & Run All*): ``/kaggle/working`` becomes the notebook output.
2. A private Kaggle Dataset (default ``aiayn-runs``) published from inside the session via
   the Kaggle CLI. It survives even an interactive session that is closed without saving.

Resuming is automatic: :func:`latest_run` finds the most advanced copy of a run anywhere
under the attached inputs (``/kaggle/input/**``), whether it came from a notebook output
or from the runs dataset.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "UPLOAD_EXCLUDES",
    "PublishResult",
    "find_runs",
    "kaggle_credentials_available",
    "latest_run",
    "publish_dataset",
    "run_progress",
    "stage_runs",
]

#: Never uploaded: the fallback copy of last.pt (local crash recovery only) and partial
#: files from interrupted writes (ADR-013).
UPLOAD_EXCLUDES: tuple[str, ...] = ("last.prev.pt", "*.tmp", "*.new", "*.part")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def run_progress(run_dir: Path) -> int:
    """Optimizer updates reached by a run, from ``metadata.json`` (0 if unknown)."""
    try:
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    updates = [s.get("end_update") or s.get("start_update") or 0 for s in meta.get("sessions", [])]
    return int(max(updates, default=0))


def _extract_zips(root: Path, into: Path) -> list[Path]:
    """Kaggle may keep uploaded folders as ``.zip`` files: extract them to search inside."""
    extracted = []
    for archive in sorted(root.rglob("*.zip")):
        target = into / archive.stem
        if not target.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        extracted.append(target)
    return extracted


def find_runs(name: str, roots: Sequence[Path], *, unzip_into: Path | None = None) -> list[Path]:
    """Every directory named ``name`` that contains a run (``config.yaml`` + ``metadata.json``)."""
    search = [Path(r) for r in roots if Path(r).exists()]
    if unzip_into is not None:
        for root in list(search):
            search += _extract_zips(root, unzip_into)
    found = {
        meta.parent
        for root in search
        for meta in root.rglob("metadata.json")
        if meta.parent.name == name and (meta.parent / "config.yaml").is_file()
    }
    return sorted(found)


def latest_run(name: str, roots: Sequence[Path], *, unzip_into: Path | None = None) -> Path | None:
    """The copy of run ``name`` with the most updates (newest file time breaks ties)."""
    runs = find_runs(name, roots, unzip_into=unzip_into)
    if not runs:
        return None
    return max(runs, key=lambda p: (run_progress(p), (p / "metadata.json").stat().st_mtime))


def _ignore(directory: str, names: list[str]) -> set[str]:
    return {n for n in names if any(fnmatch.fnmatch(n, pattern) for pattern in UPLOAD_EXCLUDES)}


def stage_runs(
    run_dirs: Sequence[Path],
    staging: Path,
    *,
    kaggle_user: str,
    slug: str = "aiayn-runs",
    previous: Path | None = None,
) -> Path:
    """Build the dataset folder: ``<staging>/runs/<name>/…`` + ``dataset-metadata.json``.

    A new dataset version **replaces** the previous content, so runs from ``previous`` (the
    attached runs dataset) are carried over unless a newer copy is being published.
    """
    if staging.exists():
        shutil.rmtree(staging)
    target = staging / "runs"
    target.mkdir(parents=True)
    names = {Path(r).name for r in run_dirs}
    if previous is not None and previous.exists():
        for old in _all_runs(previous):
            if old.name not in names:
                shutil.copytree(old, target / old.name, ignore=_ignore)
    for run in run_dirs:
        shutil.copytree(run, target / Path(run).name, ignore=_ignore)
    metadata = {
        "title": "aiayn runs",
        "id": f"{kaggle_user}/{slug}",
        "licenses": [{"name": "unknown"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    return staging


def _all_runs(root: Path) -> list[Path]:
    return sorted(
        {m.parent for m in root.rglob("metadata.json") if (m.parent / "config.yaml").is_file()}
    )


def kaggle_credentials_available() -> bool:
    """Kaggle API credentials via environment (notebook secrets) or ``kaggle.json``."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    home = Path.home()
    return any(
        (p / "kaggle.json").is_file() for p in (home / ".kaggle", home / ".config" / "kaggle")
    )


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    action: str  # "version" | "create" | "skipped"
    output: str


def publish_dataset(
    staging: Path, *, message: str, runner: Runner = subprocess.run
) -> PublishResult:
    """Upload ``staging`` as a new version of its dataset, creating it on first use.

    New datasets created through the CLI are **private** by default.
    """
    if not kaggle_credentials_available():
        return PublishResult(False, "skipped", "no Kaggle API credentials (see docs/KAGGLE.md §9)")

    def call(args: list[str]) -> Any:
        return runner(args, capture_output=True, text=True, check=False)

    version = call(
        ["kaggle", "datasets", "version", "-p", str(staging), "-m", message, "--dir-mode", "zip"]
    )
    if version.returncode == 0:
        return PublishResult(True, "version", version.stdout)
    combined = f"{version.stdout}\n{version.stderr}".lower()
    if "not found" in combined or "404" in combined or "does not exist" in combined:
        create = call(["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "zip"])
        return PublishResult(create.returncode == 0, "create", create.stdout + create.stderr)
    return PublishResult(False, "version", version.stdout + version.stderr)
