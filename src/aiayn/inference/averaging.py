"""Checkpoint averaging (paper §6.1: last 5 checkpoints for the base model; ADR-019)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

__all__ = ["average_checkpoint_files", "average_state_dicts", "load_weights", "snapshot_paths"]


def load_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Model weights from a raw state dict file or a checkpoint dict with a ``model`` key.

    Always ``weights_only=True`` on CPU.
    """
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = obj["model"] if isinstance(obj, dict) and isinstance(obj.get("model"), dict) else obj
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{path}: not a model state dict")
    return state


def average_state_dicts(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Element-wise mean of floating-point tensors (accumulated in float64 and cast back).

    Integer/boolean tensors must be identical across inputs and are copied. Keys and
    shapes must match exactly.
    """
    if not states:
        raise ValueError("nothing to average")
    keys = list(states[0])
    for i, state in enumerate(states[1:], start=1):
        if list(state) != keys:
            raise ValueError(f"state {i} has different keys")
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [s[key] for s in states]
        first = tensors[0]
        if any(t.shape != first.shape for t in tensors):
            raise ValueError(f"{key}: shapes differ")
        if first.is_floating_point():
            total = torch.zeros(first.shape, dtype=torch.float64)
            for t in tensors:
                total += t.to(torch.float64)
            averaged[key] = (total / len(tensors)).to(first.dtype)
        else:
            if any(not torch.equal(t, first) for t in tensors):
                raise ValueError(f"{key}: non-floating tensors differ and cannot be averaged")
            averaged[key] = first.clone()
    return averaged


def snapshot_paths(run_dir: Path, last: int | None = None) -> list[Path]:
    """The run's averaging snapshots (``checkpoints/avg/step_*.pt``), oldest first."""
    paths = sorted((run_dir / "checkpoints" / "avg").glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no averaging snapshots in {run_dir / 'checkpoints' / 'avg'}")
    return paths[-last:] if last else paths


def average_checkpoint_files(paths: Sequence[Path]) -> dict[str, torch.Tensor]:
    return average_state_dicts([load_weights(p) for p in paths])
