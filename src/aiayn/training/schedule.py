"""Learning-rate schedules (paper §5.3, Eq. 3; math.md §7).

``step`` is the 1-based index of the optimizer update about to be applied (ADR-017):
skipped fp16 updates do not advance it.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer

from aiayn.training.config import ScheduleConfig

__all__ = ["LRSchedule", "inverse_sqrt_lr", "noam_lr"]


def noam_lr(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    """``factor · d^-0.5 · min(step^-0.5, step · warmup^-1.5)`` (paper Eq. 3)."""
    if step < 1:
        raise ValueError(f"step counts optimizer updates from 1, got {step}")
    if warmup < 1 or d_model < 1:
        raise ValueError("warmup and d_model must be positive")
    return float(factor * d_model**-0.5 * min(step**-0.5, step * warmup**-1.5))


def inverse_sqrt_lr(step: int, peak_lr: float, warmup: int) -> float:
    """``peak · min(step / warmup, sqrt(warmup / step))``: Noam's shape with an explicit peak.

    Identical to ``noam_lr`` when ``peak = factor · (d · warmup)^-0.5`` (math.md §7.2).
    """
    if step < 1:
        raise ValueError(f"step counts optimizer updates from 1, got {step}")
    if warmup < 1:
        raise ValueError("warmup must be positive")
    return peak_lr * min(step / warmup, math.sqrt(warmup / step))


class LRSchedule:
    """Stateless schedule: the learning rate is a pure function of the update number,
    so resuming needs nothing beyond the update counter (ADR-013)."""

    def __init__(self, config: ScheduleConfig, d_model: int) -> None:
        self.config = config
        self.d_model = d_model

    def lr_at(self, step: int) -> float:
        c = self.config
        if c.kind == "noam":
            return noam_lr(step, self.d_model, c.warmup_steps, c.factor)
        if c.kind == "inverse_sqrt":
            assert c.peak_lr is not None
            return inverse_sqrt_lr(step, c.peak_lr, c.warmup_steps)
        assert c.peak_lr is not None
        return c.peak_lr  # constant

    @property
    def peak_lr(self) -> float:
        if self.config.kind == "noam":
            return self.lr_at(self.config.warmup_steps)
        assert self.config.peak_lr is not None
        return self.config.peak_lr

    def apply(self, optimizer: Optimizer, step: int) -> float:
        """Set the learning rate of every parameter group for update ``step``."""
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
