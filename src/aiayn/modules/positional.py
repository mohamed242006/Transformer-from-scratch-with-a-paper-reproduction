"""Positional encodings (paper §3.5; math.md §4).

Both variants *add* position information to ``[B, L, d]`` inputs and then apply dropout,
which is the paper's "dropout to the sums of the embeddings and the positional
encodings" (§5.4, ADR-012). An ``offset`` argument supports incremental decoding later.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from aiayn.modules._validation import check_dropout, check_last_dim, check_positive, check_rank

__all__ = ["LearnedPositionalEmbedding", "SinusoidalPositionalEncoding", "sinusoidal_table"]


def sinusoidal_table(
    max_len: int, d_model: int, *, base: float = 10000.0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return ``PE [max_len, d_model]`` with interleaved sin (even) / cos (odd) columns.

    ``PE[p, 2i] = sin(p * w_i)``, ``PE[p, 2i+1] = cos(p * w_i)``, ``w_i = base^(-2i/d)``.
    Computed in float64 (frequencies via ``exp`` of logs) and cast to ``dtype``, so large
    positions do not lose precision before the cast.
    """
    check_positive("max_len", max_len)
    check_positive("d_model", d_model)
    if d_model % 2:
        raise ValueError(f"d_model must be even for sin/cos pairs, got {d_model}")
    positions = torch.arange(max_len, dtype=torch.float64)[:, None]  # [L, 1]
    two_i = torch.arange(0, d_model, 2, dtype=torch.float64)  # [d/2] = 0, 2, 4, ...
    frequencies = torch.exp(-math.log(base) * two_i / d_model)  # w_i
    angles = positions * frequencies  # [L, d/2]
    table = torch.empty(max_len, d_model, dtype=torch.float64)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table.to(dtype)


def _check_input(x: torch.Tensor, offset: int, d_model: int, max_len: int) -> int:
    check_rank(x, 3, "x", "[B, L, d]")
    check_last_dim(x, d_model, "x")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    length = x.shape[1]
    if offset + length > max_len:
        raise ValueError(
            f"positions {offset}..{offset + length - 1} exceed max_len={max_len}; "
            "increase max_len (it must cover the longest source + 50 for inference)"
        )
    return int(length)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding + dropout. ``[B, L, d] -> [B, L, d]``.

    The table is a **non-persistent buffer**: it is not a parameter (not trained), it
    moves with ``.to(device)``, and it is excluded from ``state_dict`` because it is a
    deterministic function of ``(max_len, d)``. Checkpoints stay smaller and remain
    loadable if ``max_len`` changes.
    """

    table: torch.Tensor

    def __init__(self, d_model: int, max_len: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        check_dropout("dropout", dropout)
        self.d_model = d_model
        self.max_len = max_len
        # Stored in float64 and cast per forward: model.double() cannot recover precision
        # lost by an earlier float32 table, which would bias float64 checks (Week 4 finding).
        table = sinusoidal_table(max_len, d_model, dtype=torch.float64)
        self.register_buffer("table", table, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        length = _check_input(x, offset, self.d_model, self.max_len)
        pe = self.table[offset : offset + length].to(dtype=x.dtype)  # [L, d], broadcast over B
        out: torch.Tensor = self.dropout(x + pe)
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_len={self.max_len}"


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positions (paper Table 3 row E ablation). ``[B, L, d] -> [B, L, d]``.

    Initialized ``N(0, 1/d)`` (ADR-011). Cannot extrapolate beyond ``max_len``.
    """

    def __init__(self, d_model: int, max_len: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        check_positive("d_model", d_model)
        check_positive("max_len", max_len)
        check_dropout("dropout", dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.weight = nn.Parameter(torch.empty(max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.d_model**-0.5)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        length = _check_input(x, offset, self.d_model, self.max_len)
        out: torch.Tensor = self.dropout(x + self.weight[offset : offset + length].to(x.dtype))
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_len={self.max_len}"
