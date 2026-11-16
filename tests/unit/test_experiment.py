"""Experiment composition, compatibility and the resume-safe hash (ADR-018)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiayn.training.config import TrainConfig, load_train_config
from aiayn.training.experiment import (
    MUTABLE_ON_RESUME,
    check_compatibility,
    experiment_from_dict,
    load_experiment,
)
from aiayn.utils.config import ConfigError
from tests.conftest import REPO_ROOT

EXPERIMENTS = sorted((REPO_ROOT / "configs" / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs" / "train").glob("*.yaml")))
def test_repository_train_configs_are_valid(path: Path) -> None:
    assert load_train_config(path).name == path.stem


@pytest.mark.parametrize("path", EXPERIMENTS)
def test_repository_experiments_load_and_are_compatible(path: Path) -> None:
    experiment = load_experiment(path)
    assert experiment.name == path.stem
    check_compatibility(experiment)


def test_file_and_cli_overrides_apply_in_order(tmp_path: Path) -> None:
    exp = load_experiment(
        REPO_ROOT / "configs" / "experiment" / "debug_multi30k_cpu.yaml",
        ["train.seed=7", "data.batching.max_tokens=2048"],
    )
    assert exp.train.seed == 7
    assert exp.data.batching.max_tokens == 2048  # CLI wins over the file's 1024


def test_overrides_must_name_a_group() -> None:
    with pytest.raises(ConfigError, match="must start with"):
        load_experiment(EXPERIMENTS[0], ["seed=3"])


def test_positional_coverage_is_checked() -> None:
    exp = load_experiment(
        REPO_ROOT / "configs" / "experiment" / "multi30k_tiny.yaml", ["model.max_positions=120"]
    )
    with pytest.raises(ConfigError, match="max_positions"):
        check_compatibility(exp)  # 100 + 50 > 120


def test_token_budget_must_fit_the_longest_sequence() -> None:
    exp = load_experiment(
        REPO_ROOT / "configs" / "experiment" / "multi30k_tiny.yaml", ["data.batching.max_tokens=64"]
    )
    with pytest.raises(ConfigError, match="max_tokens"):
        check_compatibility(exp)


def test_hash_ignores_exactly_the_resume_mutable_keys() -> None:
    base = load_experiment(EXPERIMENTS[0])
    raw = base.to_dict()
    for key, value in (("max_updates", 99_999), ("time_budget_minutes", 5.0), ("log_every", 7)):
        assert key in MUTABLE_ON_RESUME
        assert experiment_from_dict(raw, [f"train.{key}={value}"]).config_hash == base.config_hash
    for override in (
        "train.label_smoothing=0.2",
        "train.seed=9",
        "model.dropout=0.3",
        "data.batching.max_tokens=1000",
    ):
        assert experiment_from_dict(raw, [override]).config_hash != base.config_hash


def test_round_trip_through_resolved_dict() -> None:
    exp = load_experiment(EXPERIMENTS[0])
    assert experiment_from_dict(exp.to_dict()) == exp


def test_train_config_validation() -> None:
    with pytest.raises(ValueError, match="max_updates"):
        TrainConfig(name="x")
    with pytest.raises(ValueError, match="loggers"):
        TrainConfig(name="x", max_updates=1, loggers=("slack",))
    with pytest.raises(ConfigError, match="not one of"):
        load_train_config(REPO_ROOT / "configs" / "train" / "debug_cpu.yaml", ("amp=false",))
