"""Cheap, sync-free argument checks shared by modules.

These inspect only shapes and dtypes (host-side metadata), never tensor values, so
they never force a GPU synchronization and are safe on the hot path.
"""

from __future__ import annotations

import torch

__all__ = ["check_dropout", "check_floating", "check_last_dim", "check_positive", "check_rank"]


def check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def check_dropout(name: str, p: float) -> None:
    if not 0.0 <= p < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {p}")


def check_rank(x: torch.Tensor, rank: int, name: str, layout: str) -> None:
    if x.dim() != rank:
        raise ValueError(f"{name} must have shape {layout}, got {tuple(x.shape)}")


def check_last_dim(x: torch.Tensor, size: int, name: str) -> None:
    if x.shape[-1] != size:
        raise ValueError(f"{name} last dimension must be {size}, got {tuple(x.shape)}")


def check_floating(x: torch.Tensor, name: str) -> None:
    if not x.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor, got {x.dtype}")
