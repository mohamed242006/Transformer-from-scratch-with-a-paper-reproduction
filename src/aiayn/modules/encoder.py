"""Encoder layer and stack (paper §3.1; ADR-015).

``Encoder`` is a pure layer stack (plus the Pre-LN final norm). It mirrors
``torch.nn.TransformerEncoder(norm=None | LayerNorm)`` so the two can be parity-tested.
Token embedding, positional encoding and mask construction belong to the model.
"""

from __future__ import annotations

import torch
from torch import nn

from aiayn.modules._validation import check_last_dim, check_positive, check_rank
from aiayn.modules.feedforward import PositionwiseFeedForward
from aiayn.modules.layernorm import LayerNorm
from aiayn.modules.multihead import MultiHeadAttention
from aiayn.modules.sublayer import NormPosition, SublayerConnection

__all__ = ["Encoder", "EncoderLayer"]


class EncoderLayer(nn.Module):
    """Self-attention sub-layer, then FFN sub-layer, each wrapped in Add & Norm.

    Shapes:
        x ``[B, S, d]``, mask ``[B, 1, 1, S]`` bool (True = may attend; ADR-008) ->
        ``(output [B, S, d], self-attention weights [B, h, S, S])`` (pre-dropout).

    Dropout sites (ADR-012): ``dropout`` = residual, ``attention_dropout`` = attention
    weights, ``activation_dropout`` = FFN inner activation.

    Parameters: ``4d² + 2·d·d_ff + d_ff + 9d`` (math.md §9.2).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        norm_position: NormPosition = "post",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=attention_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=activation_dropout)
        self.self_attn_block = SublayerConnection(
            d_model, dropout=dropout, norm_position=norm_position, eps=eps
        )
        self.ffn_block = SublayerConnection(
            d_model, dropout=dropout, norm_position=norm_position, eps=eps
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights: list[torch.Tensor] = []

        def self_attention(h: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor
            out, w = self.self_attn(h, h, h, mask)
            weights.append(w)
            return out

        x = self.self_attn_block(x, self_attention)
        x = self.ffn_block(x, self.ffn)
        return x, weights[0]


class Encoder(nn.Module):
    """``N`` independently initialized encoder layers (+ final LayerNorm when Pre-LN).

    Shapes:
        x ``[B, S, d]`` (embedded + positions), mask ``[B, 1, 1, S]`` ->
        ``(memory [B, S, d], attention)`` where ``attention`` is a tuple of ``N`` detached
        ``[B, h, S, S]`` tensors if ``need_weights`` else ``()``.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        norm_position: NormPosition = "post",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        check_positive("num_layers", num_layers)
        self.d_model = d_model
        self.norm_position: NormPosition = norm_position
        # Built one by one (not deep-copied): every layer gets its own initialization.
        self.layers = nn.ModuleList(
            EncoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                norm_position=norm_position,
                eps=eps,
            )
            for _ in range(num_layers)
        )
        self.final_norm = LayerNorm(d_model, eps=eps) if norm_position == "pre" else None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        check_rank(x, 3, "x", "[B, S, d]")
        check_last_dim(x, self.d_model, "x")
        maps: list[torch.Tensor] = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            if need_weights:
                maps.append(weights.detach())
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x, tuple(maps)
