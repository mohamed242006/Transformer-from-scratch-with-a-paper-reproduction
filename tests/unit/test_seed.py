from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from aiayn.utils.paths import find_repo_root, resolve_path
from aiayn.utils.seed import seed_everything
from tests.conftest import REPO_ROOT


def draw() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123)
    first = draw()
    seed_everything(123)
    assert draw() == first
    seed_everything(124)
    assert draw() != first


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_repo_root_and_path_resolution(tmp_path: object) -> None:
    assert find_repo_root() == REPO_ROOT
    assert resolve_path("configs") == REPO_ROOT / "configs"
    absolute = REPO_ROOT / "x"
    assert resolve_path(absolute) == absolute
