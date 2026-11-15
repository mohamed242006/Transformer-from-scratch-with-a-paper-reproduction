"""Multi-head attention (paper §3.2.2; math.md §3).

    MultiHead(X, Y) = Concat(head_1, ..., head_h) W^O,
    head_i = Attention(X W_i^Q, Y W_i^K, Y W_i^V)

The h per-head projections are stored stacked as one ``d × d`` linear map per role (Q, K,
V), so each projection is one matmul followed by a reshape into heads (math.md §3.2).
"""

from __future__ import annotations

import torch
from torch import nn

from aiayn.modules._validation import check_dropout, check_last_dim, check_positive, check_rank
from aiayn.modules.attention import scaled_dot_product_attention
from aiayn.modules.init import init_linear_

__all__ = ["MultiHeadAttention", "merge_heads", "split_heads"]


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``[B, T, h·d_h] -> [B, h, T, d_h]``; head i receives columns ``[i·d_h, (i+1)·d_h)``.

    Reshaping straight to ``[B, h, T, d_h]`` without the transpose would give the right
    shape with scrambled data (positions mixed across heads); see math.md §3.2.
    """
    batch, length, width = x.shape
    if width % num_heads:
        raise ValueError(f"width {width} is not divisible by num_heads {num_heads}")
    return x.view(batch, length, num_heads, width // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``[B, h, T, d_h] -> [B, T, h·d_h]`` (exact inverse of :func:`split_heads`)."""
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, length, heads * head_dim)


class MultiHeadAttention(nn.Module):
    """Multi-head attention for self-attention (``key = value = query``) and cross-attention.

    Shapes:
        query ``[B, Tq, d]``, key/value ``[B, Tk, d]``, mask bool broadcastable to
        ``[B, h, Tq, Tk]`` (True = may attend; ADR-008) ->
        ``(output [B, Tq, d], weights [B, h, Tq, Tk])`` (weights are pre-dropout).

    Args:
        d_model: Model width ``d``; must be divisible by ``num_heads``.
        num_heads: Number of heads ``h``; ``d_k = d_v = d / h``.
        dropout: Attention-weight dropout (ADR-012; default 0.0, paper-literal for MT).
        bias: Use biases in the four projections (paper/reference implementations: yes).

    Parameters: ``4d² + 4d`` with bias (math.md §9.1). Init: Xavier-uniform, zero bias.
    """

    def __init__(
        self, d_model: int, num_heads: int, *, dropout: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        check_positive("d_model", d_model)
        check_positive("num_heads", num_heads)
        check_dropout("dropout", dropout)
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            init_linear_(layer)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for name, t in (("query", query), ("key", key), ("value", value)):
            check_rank(t, 3, name, "[B, T, d]")
            check_last_dim(t, self.d_model, name)
        if not query.shape[0] == key.shape[0] == value.shape[0]:
            raise ValueError("query, key and value must share the batch dimension")
        if key.shape[1] != value.shape[1]:
            raise ValueError(f"key/value lengths differ: {key.shape[1]} vs {value.shape[1]}")

        q = split_heads(self.q_proj(query), self.num_heads)  # [B, h, Tq, d_h]
        k = split_heads(self.k_proj(key), self.num_heads)  # [B, h, Tk, d_h]
        v = split_heads(self.v_proj(value), self.num_heads)  # [B, h, Tk, d_h]
        context, weights = scaled_dot_product_attention(
            q, k, v, mask, dropout_p=self.dropout, training=self.training
        )  # [B, h, Tq, d_h], [B, h, Tq, Tk]
        return self.out_proj(merge_heads(context)), weights  # [B, Tq, d]

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, dropout={self.dropout}"
