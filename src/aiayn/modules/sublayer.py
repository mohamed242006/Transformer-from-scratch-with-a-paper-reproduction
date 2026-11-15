"""Residual connection + dropout + LayerNorm around a sub-layer (paper §3.1, §5.4; ADR-003).

    post (paper, default):  LN(x + Dropout(Sublayer(x)))
    pre  (ablation):        x + Dropout(Sublayer(LN(x)))     (stack needs a final LN; Week 4)

Dropout here is the paper's *residual dropout*: applied to the sub-layer output before it
is added to the sub-layer input (ADR-012).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from aiayn.modules._validation import check_dropout
from aiayn.modules.layernorm import LayerNorm

__all__ = ["NormPosition", "SublayerConnection", "SublayerFn"]

NormPosition = Literal["post", "pre"]
SublayerFn = Callable[[torch.Tensor], torch.Tensor]


class SublayerConnection(nn.Module):
    """Wraps any shape-preserving sub-layer ``f: [B, T, d] -> [B, T, d]``.

    Usage (Week 4)::

        x = self.self_attn_block(x, lambda h: self.self_attn(h, h, h, mask)[0])
        x = self.ffn_block(x, self.ffn)

    Args:
        d_model: Feature size ``d`` (for the LayerNorm).
        dropout: Residual dropout ``P_drop``.
        norm_position: ``"post"`` (paper-exact, default) or ``"pre"``.
        eps: LayerNorm epsilon.
    """

    def __init__(
        self,
        d_model: int,
        *,
        dropout: float = 0.0,
        norm_position: NormPosition = "post",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        check_dropout("dropout", dropout)
        if norm_position not in ("post", "pre"):
            raise ValueError(f"norm_position must be 'post' or 'pre', got {norm_position!r}")
        self.norm_position: NormPosition = norm_position
        self.norm = LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: SublayerFn) -> torch.Tensor:
        out: torch.Tensor
        if self.norm_position == "post":
            out = self.norm(x + self.dropout(self._run_sublayer(sublayer, x)))
        else:
            out = x + self.dropout(self._run_sublayer(sublayer, self.norm(x)))
        return out

    @staticmethod
    def _run_sublayer(
        sublayer: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        out = sublayer(x)
        if out.shape != x.shape:
            raise ValueError(
                f"sub-layer must preserve shape for the residual add: "
                f"{tuple(x.shape)} -> {tuple(out.shape)}"
            )
        return out

    def extra_repr(self) -> str:
        return f"norm_position={self.norm_position!r}"
