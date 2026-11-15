"""Parameter counting that respects shared (tied) parameters."""

from __future__ import annotations

from collections import defaultdict

from torch import nn

__all__ = ["count_parameters", "parameter_breakdown"]


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Number of **unique** parameters (a tied matrix is counted once)."""
    seen: set[int] = set()
    total = 0
    for p in module.parameters():  # parameters() already de-duplicates shared tensors
        if (trainable_only and not p.requires_grad) or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def parameter_breakdown(module: nn.Module) -> dict[str, int]:
    """Unique parameter counts per top-level child (first-registered owner wins)."""
    counts: dict[str, int] = defaultdict(int)
    for name, p in module.named_parameters():  # de-duplicated, registration order
        counts[name.split(".", 1)[0]] += p.numel()
    return dict(counts)
