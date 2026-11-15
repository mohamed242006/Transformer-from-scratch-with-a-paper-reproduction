"""Position-wise feed-forward network (paper §3.3, Eq. 2).

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

Applied to every position independently ("two convolutions with kernel size 1").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from aiayn.modules._validation import check_dropout, check_last_dim, check_positive
from aiayn.modules.init import init_linear_

__all__ = ["PositionwiseFeedForward"]


class PositionwiseFeedForward(nn.Module):
    """``[..., d] -> [..., d_ff] -> ReLU -> (dropout) -> [..., d]``.

    Args:
        d_model: Input/output width ``d``.
        d_ff: Inner width (paper base: 2048; IWSLT config: 1024).
        dropout: Dropout on the inner activation (ADR-012: default 0.0, paper silent).
        bias: Use biases ``b_1``, ``b_2`` (paper: yes).

    Parameters: ``2·d·d_ff + d_ff + d`` (math.md §9.1). Init: Xavier-uniform, zero bias.
    """

    def __init__(self, d_model: int, d_ff: int, *, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        check_positive("d_model", d_model)
        check_positive("d_ff", d_ff)
        check_dropout("dropout", dropout)
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear_(self.linear1)
        init_linear_(self.linear2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_last_dim(x, self.d_model, "x")
        out: torch.Tensor = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return out
