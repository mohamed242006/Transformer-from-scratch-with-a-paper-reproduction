"""Atomic checkpoints, fallback, snapshots and RNG capture (ADR-013)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from aiayn.training.checkpoint import (
    CheckpointError,
    atomic_save,
    capture_rng_state,
    load_training_state,
    prune_snapshots,
    restore_rng_state,
    save_training_state,
)


def state(value: float) -> dict[str, object]:
    return {"model": {"w": torch.tensor([value])}, "progress": {"update": int(value)}}


def test_rotation_keeps_previous_state(tmp_path: Path) -> None:
    save_training_state(tmp_path, state(1.0))
    save_training_state(tmp_path, state(2.0))
    loaded, path = load_training_state(tmp_path)
    assert path.name == "last.pt" and loaded["progress"] == {"update": 2}
    previous = torch.load(tmp_path / "last.prev.pt", weights_only=True)
    assert previous["progress"] == {"update": 1}
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.new"))


def test_corrupt_last_falls_back_to_previous(tmp_path: Path) -> None:
    save_training_state(tmp_path, state(1.0))
    save_training_state(tmp_path, state(2.0))
    (tmp_path / "last.pt").write_bytes(b"truncated by a killed session")
    loaded, path = load_training_state(tmp_path)
    assert path.name == "last.prev.pt" and loaded["progress"] == {"update": 1}


def test_missing_last_after_interrupted_rotation_falls_back(tmp_path: Path) -> None:
    save_training_state(tmp_path, state(1.0))
    save_training_state(tmp_path, state(2.0))
    (tmp_path / "last.pt").unlink()  # killed between rotate and rename
    assert load_training_state(tmp_path)[1].name == "last.prev.pt"


def test_no_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        load_training_state(tmp_path)


class Evil:
    """Stands in for any non-allowlisted object that could execute code when unpickled."""


def test_weights_only_rejects_arbitrary_objects(tmp_path: Path) -> None:
    torch.save({"model": {}, "payload": Evil()}, tmp_path / "last.pt")
    with pytest.raises(CheckpointError):
        load_training_state(tmp_path)


def test_atomic_save_and_snapshot_pruning(tmp_path: Path) -> None:
    for step in (5, 10, 15, 20):
        atomic_save({"w": torch.tensor([step])}, tmp_path / f"step_{step:08d}.pt")
    removed = prune_snapshots(tmp_path, keep=2)
    assert [p.name for p in removed] == ["step_00000005.pt", "step_00000010.pt"]
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == [
        "step_00000015.pt",
        "step_00000020.pt",
    ]


def test_rng_state_round_trip_through_a_weights_only_file(tmp_path: Path) -> None:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    atomic_save({"rng": capture_rng_state()}, tmp_path / "rng.pt")
    expected = (random.random(), float(np.random.rand()), torch.rand(3).tolist())
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(torch.load(tmp_path / "rng.pt", weights_only=True)["rng"])
    assert (random.random(), float(np.random.rand()), torch.rand(3).tolist()) == expected
