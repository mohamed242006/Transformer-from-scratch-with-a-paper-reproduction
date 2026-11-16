from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aiayn.inference.averaging import (
    average_checkpoint_files,
    average_state_dicts,
    load_weights,
    snapshot_paths,
)


def test_floating_tensors_are_averaged_and_dtypes_kept() -> None:
    a = {
        "w": torch.tensor([1.0, 3.0]),
        "h": torch.tensor([1.0], dtype=torch.float16),
        "count": torch.tensor([7]),
    }
    b = {
        "w": torch.tensor([3.0, 5.0]),
        "h": torch.tensor([2.0], dtype=torch.float16),
        "count": torch.tensor([7]),
    }
    out = average_state_dicts([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0]))
    assert out["h"].dtype == torch.float16 and out["h"].item() == 1.5
    assert torch.equal(out["count"], torch.tensor([7]))


def test_averaging_identical_states_is_the_identity() -> None:
    state = {"w": torch.randn(4, 4)}
    torch.testing.assert_close(average_state_dicts([state, state, state])["w"], state["w"])


@pytest.mark.parametrize(
    "other",
    [{"v": torch.zeros(2)}, {"w": torch.zeros(3)}, {"w": torch.zeros(2), "n": torch.tensor([1])}],
)
def test_mismatches_are_rejected(other: dict[str, torch.Tensor]) -> None:
    with pytest.raises(ValueError):
        average_state_dicts([{"w": torch.zeros(2)}, other])


def test_differing_integer_tensors_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-floating"):
        average_state_dicts([{"n": torch.tensor([1])}, {"n": torch.tensor([2])}])


def test_files_snapshots_and_checkpoint_dicts(tmp_path: Path) -> None:
    avg_dir = tmp_path / "checkpoints" / "avg"
    avg_dir.mkdir(parents=True)
    for step, value in ((100, 1.0), (200, 2.0), (300, 6.0)):
        torch.save({"w": torch.tensor([value])}, avg_dir / f"step_{step:08d}.pt")
    torch.save({"model": {"w": torch.tensor([9.0])}, "update": 3}, tmp_path / "best.pt")
    assert [p.name for p in snapshot_paths(tmp_path, last=2)] == [
        "step_00000200.pt",
        "step_00000300.pt",
    ]
    assert average_checkpoint_files(snapshot_paths(tmp_path))["w"].item() == pytest.approx(3.0)
    assert load_weights(tmp_path / "best.pt")["w"].item() == 9.0  # checkpoint dict form
    with pytest.raises(FileNotFoundError):
        snapshot_paths(tmp_path / "missing")
    with pytest.raises(ValueError):
        average_state_dicts([])
