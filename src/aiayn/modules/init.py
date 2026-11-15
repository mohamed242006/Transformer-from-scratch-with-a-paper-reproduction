"""Initialization helpers implementing ADR-011."""

from __future__ import annotations

from torch import nn

__all__ = ["init_linear_"]


def init_linear_(layer: nn.Linear) -> None:
    """Xavier-uniform weight, zero bias (ADR-011).

    The bound is ``sqrt(6 / (fan_in + fan_out))``, which keeps activation variance
    roughly constant in both the forward and backward pass (math.md §5.4).
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
