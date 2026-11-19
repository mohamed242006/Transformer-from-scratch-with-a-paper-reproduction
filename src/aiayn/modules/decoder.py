"""Decoder layer and stack (paper §3.1, §3.2.3; ADR-015).

``Decoder`` mirrors ``torch.nn.TransformerDecoder(norm=None | LayerNorm)``: a pure layer
stack over embedded target inputs, attending to the encoder memory.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from aiayn.modules._validation import check_last_dim, check_positive, check_rank
from aiayn.modules.feedforward import PositionwiseFeedForward
from aiayn.modules.layernorm import LayerNorm
from aiayn.modules.multihead import MultiHeadAttention
from aiayn.modules.sublayer import NormPosition, SublayerConnection

__all__ = ["Decoder", "DecoderLayer", "LayerCache"]


class LayerCache:
    """Incremental-decoding state of one decoder layer (ADR-023 §6).

    ``self_k``/``self_v``: keys/values of every target position so far ``[B, h, t, d_h]``.
    ``cross_k``/``cross_v``: projected encoder memory ``[B, h, S, d_h]`` (computed once).
    """

    def __init__(self, cross_k: torch.Tensor, cross_v: torch.Tensor) -> None:
        self.cross_k, self.cross_v = cross_k, cross_v
        self.self_k: torch.Tensor | None = None
        self.self_v: torch.Tensor | None = None

    def reorder(self, rows: torch.Tensor) -> None:
        """Keep batch rows ``rows`` (beam search reselection), in place."""
        self.cross_k, self.cross_v = self.cross_k[rows], self.cross_v[rows]
        if self.self_k is not None and self.self_v is not None:
            self.self_k, self.self_v = self.self_k[rows], self.self_v[rows]


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

    def start(self, memory: torch.Tensor) -> LayerCache:
        return LayerCache(*self.cross_attn.project_kv(memory, memory))

    def step(self, x: torch.Tensor, cache: LayerCache, memory_mask: torch.Tensor) -> torch.Tensor:
        """One new position ``x [B, 1, d]`` given ``cache`` (updated in place).

        Same sub-layer blocks as ``forward``; the new query attends to all cached
        positions, which is exactly the causal mask's row for this position.
        """

        def self_attention(h: torch.Tensor) -> torch.Tensor:
            k, v = self.self_attn.project_kv(h, h)
            if cache.self_k is not None and cache.self_v is not None:
                k = torch.cat([cache.self_k, k], dim=2)
                v = torch.cat([cache.self_v, v], dim=2)
            cache.self_k, cache.self_v = k, v
            out: torch.Tensor = self.self_attn.attend(h, k, v, None)[0]
            return out

        def cross_attention(h: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.cross_attn.attend(
                h, cache.cross_k, cache.cross_v, memory_mask
            )[0]
            return out

        x = self.self_attn_block(x, self_attention)
        x = self.cross_attn_block(x, cross_attention)
        out: torch.Tensor = self.ffn_block(x, self.ffn)
        return out


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

    def start(self, memory: torch.Tensor) -> list[LayerCache]:
        """Per-layer caches with the memory projected once (incremental decoding)."""
        return [cast(DecoderLayer, layer).start(memory) for layer in self.layers]

    def step(
        self, x: torch.Tensor, caches: list[LayerCache], memory_mask: torch.Tensor
    ) -> torch.Tensor:
        """``x [B, 1, d]`` (embedded new position) -> ``hidden [B, 1, d]``."""
        for layer, cache in zip(self.layers, caches, strict=True):
            x = cast(DecoderLayer, layer).step(x, cache, memory_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x
