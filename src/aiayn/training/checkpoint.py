"""Checkpoint files and RNG state (ADR-013).

Layout inside ``<run_dir>/checkpoints/``::

    last.pt        full training state (atomically replaced)
    last.prev.pt   the previous full state (fallback if last.pt is unreadable)
    best.pt        weights only, best validation NLL
    avg/step_XXXXXXXX.pt  weights only, sliding window for checkpoint averaging

Everything is saved with ``torch.save`` and loaded with ``weights_only=True``. NumPy and
Python RNG states are converted to tensors and plain containers so that this works.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "CheckpointError",
    "atomic_save",
    "capture_rng_state",
    "load_training_state",
    "prune_snapshots",
    "restore_rng_state",
    "save_training_state",
]

LAST = "last.pt"
PREVIOUS = "last.prev.pt"


class CheckpointError(RuntimeError):
    """No usable checkpoint could be loaded."""


def atomic_save(obj: Any, path: Path) -> Path:
    """Write ``obj`` to ``path`` so that readers never see a partial file.

    The data goes to ``<path>.tmp``, is flushed and fsync'ed, then renamed over ``path``
    (``os.replace`` is atomic on POSIX and Windows).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def save_training_state(checkpoint_dir: Path, state: dict[str, Any]) -> Path:
    """Write a full training state as ``last.pt``, keeping the old one as ``last.prev.pt``.

    Order: (1) write the new state to a temp file, (2) rotate ``last`` -> ``last.prev``,
    (3) rename the temp file to ``last``. A kill at any point leaves a loadable state.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last, previous = checkpoint_dir / LAST, checkpoint_dir / PREVIOUS
    tmp = checkpoint_dir / (LAST + ".new")
    atomic_save(state, tmp)
    if last.exists():
        os.replace(last, previous)
    os.replace(tmp, last)
    return last


def load_training_state(
    checkpoint_dir: Path, map_location: str | torch.device = "cpu"
) -> tuple[dict[str, Any], Path]:
    """Load ``last.pt``, falling back to ``last.prev.pt`` if it is missing or corrupt."""
    errors: list[str] = []
    for name in (LAST, PREVIOUS):
        path = checkpoint_dir / name
        if not path.is_file():
            errors.append(f"{name}: missing")
            continue
        try:
            state = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:  # truncated or corrupt file: try the previous one
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(state, dict) or "model" not in state:
            errors.append(f"{name}: not a training state")
            continue
        return state, path
    raise CheckpointError(f"no loadable checkpoint in {checkpoint_dir}: {errors}")


def prune_snapshots(directory: Path, keep: int) -> list[Path]:
    """Keep the ``keep`` newest ``step_*.pt`` files (names sort by step); return removed."""
    snapshots = sorted(directory.glob("step_*.pt"))
    removed = snapshots[:-keep] if keep > 0 else snapshots
    for path in removed:
        path.unlink()
    return removed


def capture_rng_state() -> dict[str, Any]:
    """Python, NumPy, torch-CPU and torch-CUDA RNG states in ``weights_only``-safe types."""
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state(legacy=True)
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "name": np_name,
            "keys": torch.from_numpy(np_keys.astype(np.int64)),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Inverse of :func:`capture_rng_state`. CUDA states are restored only if the device
    count matches (resuming on a different GPU type is allowed but not bit-exact)."""
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    np_state = state["numpy"]
    np.random.set_state(
        (
            np_state["name"],
            np_state["keys"].numpy().astype(np.uint32),
            np_state["pos"],
            np_state["has_gauss"],
            np_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda") or []
    if cuda_states and torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_states)
