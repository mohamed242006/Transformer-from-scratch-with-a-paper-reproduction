"""Decoder layer and stack (paper §3.1, §3.2.3; ADR-015).

``Decoder`` mirrors ``torch.nn.TransformerDecoder(norm=None | LayerNorm)``: a pure layer
stack over embedded target inputs, attending to the encoder memory.
"""

from __future__ import annotations

import torch
from torch import nn

from aiayn.modules._validation import check_last_dim, check_positive, check_rank
from aiayn.modules.feedforward import PositionwiseFeedForward
from aiayn.modules.layernorm import LayerNorm
from aiayn.modules.multihead import MultiHeadAttention
from aiayn.modules.sublayer import NormPosition, SublayerConnection

__all__ = ["Decoder", "DecoderLayer"]


class DecoderLayer(nn.Module):
    """Masked self-attention, encoder-decoder (cross) attention, FFN; each with Add & Norm.

    Shapes:
        x ``[B, T, d]``, memory ``[B, S, d]``,
        self_mask ``[B, 1, T, T]`` (causal AND target padding),
        memory_mask ``[B, 1, 1, S]`` (source padding) ->
        ``(output [B, T, d], self weights [B, h, T, T], cross weights [B, h, T, S])``.

    In cross-attention the **queries come from the decoder** and keys/values from the
    memory (paper §3.2.3). The memory mask is the *source* padding mask, never the target
    mask.

    Parameters: ``8d² + 2·d·d_ff + d_ff + 15d`` (math.md §9.2).
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
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=attention_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=activation_dropout)
        blocks = [
            SublayerConnection(d_model, dropout=dropout, norm_position=norm_position, eps=eps)
            for _ in range(3)
        ]
        self.self_attn_block, self.cross_attn_block, self.ffn_block = blocks

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_weights: list[torch.Tensor] = []
        cross_weights: list[torch.Tensor] = []

        def self_attention(h: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor
            out, w = self.self_attn(h, h, h, self_mask)
            self_weights.append(w)
            return out

        def cross_attention(h: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor
            out, w = self.cross_attn(h, memory, memory, memory_mask)
            cross_weights.append(w)
            return out

        x = self.self_attn_block(x, self_attention)
        x = self.cross_attn_block(x, cross_attention)
        x = self.ffn_block(x, self.ffn)
        return x, self_weights[0], cross_weights[0]


class Decoder(nn.Module):
    """``N`` independently initialized decoder layers (+ final LayerNorm when Pre-LN).

    Shapes:
        x ``[B, T, d]``, memory ``[B, S, d]``, self_mask ``[B, 1, T, T]``,
        memory_mask ``[B, 1, 1, S]`` -> ``(hidden [B, T, d], self_maps, cross_maps)``
        where the maps are tuples of ``N`` detached tensors if ``need_weights`` else ``()``.
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
        self.layers = nn.ModuleList(
            DecoderLayer(
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
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        check_rank(x, 3, "x", "[B, T, d]")
        check_rank(memory, 3, "memory", "[B, S, d]")
        check_last_dim(x, self.d_model, "x")
        check_last_dim(memory, self.d_model, "memory")
        if x.shape[0] != memory.shape[0]:
            raise ValueError(f"batch mismatch: x {x.shape[0]} vs memory {memory.shape[0]}")
        self_maps: list[torch.Tensor] = []
        cross_maps: list[torch.Tensor] = []
        for layer in self.layers:
            x, self_w, cross_w = layer(x, memory, self_mask, memory_mask)
            if need_weights:
                self_maps.append(self_w.detach())
                cross_maps.append(cross_w.detach())
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x, tuple(self_maps), tuple(cross_maps)
