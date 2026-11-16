"""Training configuration schema (``configs/train/*.yaml``; ADR-007, ADR-017)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from aiayn.utils.config import load_config

__all__ = ["OptimizerConfig", "ScheduleConfig", "TrainConfig", "load_train_config"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Adam (paper §5.3: β1 = 0.9, β2 = 0.98, ε = 1e-9, no weight decay)."""

    betas: tuple[float, ...] = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if len(self.betas) != 2 or not all(0.0 <= b < 1.0 for b in self.betas):
            raise ValueError(f"betas must be two values in [0, 1), got {self.betas}")
        if self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("eps must be positive and weight_decay non-negative")


@dataclass(frozen=True)
class ScheduleConfig:
    """``noam``: paper Eq. 3 × ``factor``. ``inverse_sqrt``: same shape, explicit ``peak_lr``.
    ``constant``: ``peak_lr`` throughout (debugging only)."""

    kind: Literal["noam", "inverse_sqrt", "constant"] = "noam"
    warmup_steps: int = 4000
    factor: float = 1.0
    peak_lr: float | None = None

    def __post_init__(self) -> None:
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        if self.kind == "noam" and self.peak_lr is not None:
            raise ValueError("noam derives its peak from d_model and warmup; use inverse_sqrt")
        if self.kind != "noam" and (self.peak_lr is None or self.peak_lr <= 0):
            raise ValueError(f"schedule '{self.kind}' requires a positive peak_lr")
        if self.factor <= 0:
            raise ValueError("factor must be positive")


@dataclass(frozen=True)
class TrainConfig:
    """Optimization and run-control settings. ``max_updates``/``max_epochs`` of 0 = no limit
    (at least one must be set)."""

    name: str
    seed: int = 1
    max_updates: int = 0
    max_epochs: int = 0
    update_freq: int = 1
    label_smoothing: float = 0.1
    clip_norm: float = 0.0
    amp: Literal["auto", "fp16", "bf16", "fp32"] = "auto"  # not "off": YAML reads it as False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    log_every: int = 50
    validate_every: int = 0
    save_every: int = 1000
    average_last: int = 5
    time_budget_minutes: float = 0.0
    loggers: tuple[str, ...] = ("stdout", "jsonl")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not re.match(r"^[a-z0-9_]+$", self.name):
            raise ValueError(f"name must be lowercase snake_case: {self.name!r}")
        if (
            self.max_updates < 0
            or self.max_epochs < 0
            or (self.max_updates == self.max_epochs == 0)
        ):
            raise ValueError("set max_updates and/or max_epochs (non-negative, not both 0)")
        if self.seed < 0 or self.update_freq < 1:
            raise ValueError("seed must be >= 0 and update_freq >= 1")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.clip_norm < 0 or self.time_budget_minutes < 0:
            raise ValueError("clip_norm and time_budget_minutes must be non-negative")
        if self.log_every < 1 or self.validate_every < 0 or self.save_every < 0:
            raise ValueError("log_every >= 1; validate_every and save_every >= 0")
        if self.average_last < 1:
            raise ValueError("average_last must be >= 1")
        unknown = set(self.loggers) - {"stdout", "jsonl", "tensorboard", "wandb"}
        if unknown:
            raise ValueError(f"unknown loggers: {sorted(unknown)}")


def load_train_config(path: str | Path, overrides: tuple[str, ...] = ()) -> TrainConfig:
    return load_config(path, TrainConfig, overrides)
