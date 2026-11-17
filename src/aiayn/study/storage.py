"""Run state as seen by a study, and slimming of completed runs (ADR-022).

A study produces ~100 runs; a full run directory (``last.pt`` with optimizer state,
``last.prev.pt``, ``best.pt``, five averaging snapshots) is ~330 MB for the 7.6M-parameter
models, which would exceed Kaggle's 20 GB output limit. A **completed** run is therefore
slimmed to what evaluation and analysis need:

* ``checkpoints/averaged.pt``: mean of the averaging snapshots (weights only);
* ``checkpoints/best.pt``: weights only, kept for runs that are evaluated on "best";
* config, metadata, metrics, tokenizer and ``eval/`` stay untouched.

Slimmed runs cannot be resumed (they are complete) and stay fully evaluable.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from aiayn.inference.averaging import average_checkpoint_files, load_weights, snapshot_paths

__all__ = ["FAILED_MARKER", "RunState", "mark_failed", "read_state", "slim_run"]

FAILED_MARKER = "FAILED.json"


@dataclass(frozen=True)
class RunState:
    path: Path
    completed: bool
    failed: bool
    has_checkpoints: bool
    best_valid_nll: float | None
    parameters: int | None
    evals: dict[str, dict[str, Any]] = field(default_factory=dict)  # tag -> eval JSON
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.completed and not self.failed


def read_state(run_dir: Path) -> RunState:
    meta: dict[str, Any] = {}
    meta_path = run_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sessions = meta.get("sessions", [])
    evals = {}
    if (run_dir / "eval").is_dir():
        for path in sorted((run_dir / "eval").glob("*.json")):
            evals[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    ckpt = run_dir / "checkpoints"
    return RunState(
        path=run_dir,
        completed=bool(sessions) and sessions[-1].get("status") == "completed",
        failed=(run_dir / FAILED_MARKER).is_file(),
        has_checkpoints=(ckpt / "last.pt").is_file() or (ckpt / "last.prev.pt").is_file(),
        best_valid_nll=meta.get("best_valid_nll"),
        parameters=meta.get("parameters"),
        evals=evals,
        metadata=meta,
    )


def mark_failed(run_dir: Path, returncode: int, detail: str = "") -> None:
    """Record a run that crashed or diverged; the study reports it and moves on."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"returncode": returncode, "detail": detail[-2000:]}
    (run_dir / FAILED_MARKER).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def slim_run(run_dir: Path, *, keep_best: bool) -> int:
    """Reduce a completed run to evaluation essentials; returns bytes freed. Idempotent."""
    state = read_state(run_dir)
    if not state.completed:
        raise ValueError(f"{run_dir}: only completed runs can be slimmed")
    ckpt = run_dir / "checkpoints"
    before = sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file())
    if not (ckpt / "averaged.pt").is_file():
        _atomic_save(average_checkpoint_files(snapshot_paths(run_dir)), ckpt / "averaged.pt")
    best = ckpt / "best.pt"
    if best.is_file():
        if keep_best:
            raw = torch.load(best, map_location="cpu", weights_only=True)
            if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
                _atomic_save(load_weights(best), best)  # drop optimizer/RNG state
        else:
            best.unlink()
    for name in ("last.pt", "last.prev.pt"):
        (ckpt / name).unlink(missing_ok=True)
    shutil.rmtree(ckpt / "avg", ignore_errors=True)
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("slimmed"):
        meta["slimmed"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    after = sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file())
    return before - after
