"""Experiment composition: data + model + train groups (ADR-018)."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiayn.data.config import DataConfig
from aiayn.models.config import ModelConfig
from aiayn.training.config import TrainConfig
from aiayn.utils.config import (
    ConfigError,
    apply_overrides,
    config_to_dict,
    from_dict,
    load_yaml,
)
from aiayn.utils.hashing import sha256_bytes
from aiayn.utils.paths import find_repo_root, resolve_path

__all__ = [
    "GROUPS",
    "MUTABLE_ON_RESUME",
    "Experiment",
    "check_compatibility",
    "experiment_from_dict",
    "load_experiment",
]

GROUPS = ("data", "model", "train")

#: Keys a resume may change without changing the experiment's identity (ADR-018 §4).
MUTABLE_ON_RESUME: tuple[str, ...] = (
    "max_updates",
    "max_epochs",
    "time_budget_minutes",
    "log_every",
    "loggers",
    "device",
)

#: Inference may decode up to source length + 50 (paper §6.1).
DECODE_MARGIN = 50


@dataclass(frozen=True)
class Experiment:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data": config_to_dict(self.data),
            "model": config_to_dict(self.model),
            "train": config_to_dict(self.train),
        }

    @property
    def config_hash(self) -> str:
        """Identity of the experiment, ignoring resume-mutable training keys."""
        resolved = self.to_dict()
        for key in MUTABLE_ON_RESUME:
            resolved["train"].pop(key, None)
        return sha256_bytes(json.dumps(resolved, sort_keys=True).encode("utf-8"))


def _group_file(group: str, value: str, base: Path) -> Path:
    if value.endswith((".yaml", ".yml")) or "/" in value or "\\" in value:
        return resolve_path(value, base)
    return base / "configs" / group / f"{value}.yaml"


def _split_overrides(overrides: Sequence[str]) -> dict[str, list[str]]:
    split: dict[str, list[str]] = {g: [] for g in GROUPS}
    for item in overrides:
        group, _, rest = item.partition(".")
        if group not in GROUPS or not rest:
            raise ConfigError(
                f"override {item!r} must start with one of {GROUPS}, e.g. train.seed=2"
            )
        split[group].append(rest)
    return split


def experiment_from_dict(raw: dict[str, Any], overrides: Sequence[str] = ()) -> Experiment:
    """Build from a resolved mapping (e.g. a run's ``config.yaml``) plus group overrides."""
    unknown = set(raw) - {"name", *GROUPS}
    if unknown or "name" not in raw:
        raise ConfigError(f"experiment needs 'name', {GROUPS}; unexpected keys: {sorted(unknown)}")
    split = _split_overrides(overrides)
    groups = {g: apply_overrides(raw[g], split[g]) for g in GROUPS}
    try:
        return Experiment(
            name=str(raw["name"]),
            data=from_dict(DataConfig, groups["data"], "data"),
            model=from_dict(ModelConfig, groups["model"], "model"),
            train=from_dict(TrainConfig, groups["train"], "train"),
        )
    except ConfigError as exc:
        raise ConfigError(f"experiment {raw.get('name')!r}: {exc}") from exc


def load_experiment(path: str | Path, overrides: Sequence[str] = ()) -> Experiment:
    """Load ``configs/experiment/*.yaml``: resolve group files, apply file then CLI overrides."""
    path = Path(path)
    raw = load_yaml(path)
    allowed = {"name", *GROUPS, "overrides"}
    if set(raw) - allowed or not {"name", *GROUPS} <= set(raw):
        raise ConfigError(f"{path}: experiment keys must be {sorted(allowed)}")
    base = find_repo_root(path.resolve())
    resolved: dict[str, Any] = {"name": raw["name"]}
    for group in GROUPS:
        value = raw[group]
        if not isinstance(value, str):
            raise ConfigError(f"{path}: '{group}' must name a config file")
        resolved[group] = load_yaml(_group_file(group, value, base))
    file_overrides = raw.get("overrides") or []
    if not isinstance(file_overrides, list):
        raise ConfigError(f"{path}: 'overrides' must be a list of 'group.key=value' strings")
    try:
        return experiment_from_dict(copy.deepcopy(resolved), [*file_overrides, *overrides])
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def max_sequence_length(data: DataConfig) -> int:
    """Longest tensor length (S or T) the data pipeline can produce."""
    if data.translation is not None:
        return data.translation.filter.max_len
    assert data.copy is not None
    return data.copy.max_len + 1


def check_compatibility(experiment: Experiment) -> None:
    """Cross-group checks that no single config can make (ADR-018 §3)."""
    longest = max_sequence_length(experiment.data)
    needed = longest + DECODE_MARGIN
    if experiment.model.max_positions < needed:
        raise ConfigError(
            f"model.max_positions={experiment.model.max_positions} < {needed} "
            f"(longest sequence {longest} + {DECODE_MARGIN} for decoding)"
        )
    for name, budget in (
        ("max_tokens", experiment.data.batching.max_tokens),
        ("eval_max_tokens", experiment.data.batching.eval_max_tokens or longest),
    ):
        if budget < longest:
            raise ConfigError(f"data.batching.{name}={budget} < longest sequence {longest}")
