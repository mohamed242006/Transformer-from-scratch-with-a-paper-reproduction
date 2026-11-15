"""The full encoder-decoder Transformer (paper §3; ADR-015).

Data flow (shapes; B batch, S source length, T decoder length, d width, V vocabulary)::

    src [B,S] ─ embed·√d ─ +PE ─ dropout ─ Encoder ──────────────► memory [B,S,d]
    tgt_in [B,T] ─ embed·√d ─ +PE ─ dropout ─ Decoder(memory) ─► hidden [B,T,d]
    hidden ─ h·Eᵀ (tied, no bias) ─► logits [B,T,V]

Masks are built here from ``pad_id`` (ADR-008): source ``[B,1,1,S]`` for encoder
self-attention and cross-attention; decoder ``[B,1,T,T]`` = causal ∧ target padding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from aiayn.models.config import ModelConfig
from aiayn.modules.decoder import Decoder
from aiayn.modules.embedding import TokenEmbedding
from aiayn.modules.encoder import Encoder
from aiayn.modules.feedforward import PositionwiseFeedForward
from aiayn.modules.layernorm import LayerNorm
from aiayn.modules.masking import (
    assert_valid_attention_mask,
    decoder_attention_mask,
    encoder_attention_mask,
)
from aiayn.modules.multihead import MultiHeadAttention
from aiayn.modules.positional import LearnedPositionalEmbedding, SinusoidalPositionalEncoding

__all__ = ["AttentionMaps", "ModelOutput", "Transformer", "build_transformer"]

Positional = SinusoidalPositionalEncoding | LearnedPositionalEmbedding

# Modules whose reset_parameters() implements ADR-011. Raw nn.Linear is deliberately
# absent: its default (Kaiming) reset would undo the Xavier initialization.
_SELF_INITIALIZING = (
    TokenEmbedding,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    LayerNorm,
)


@dataclass(frozen=True)
class AttentionMaps:
    """Detached, pre-dropout attention weights, one tensor per layer (ADR-015 §5).

    encoder_self ``[B, h, S, S]`` · decoder_self ``[B, h, T, T]`` · decoder_cross ``[B, h, T, S]``
    """

    encoder_self: tuple[torch.Tensor, ...]
    decoder_self: tuple[torch.Tensor, ...]
    decoder_cross: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    attention: AttentionMaps


def _make_positional(config: ModelConfig) -> Positional:
    if config.positional_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(
            config.d_model, config.max_positions, dropout=config.dropout
        )
    return LearnedPositionalEmbedding(config.d_model, config.max_positions, dropout=config.dropout)


class Transformer(nn.Module):
    """Encoder-decoder Transformer built only from ``aiayn.modules`` components.

    Args:
        config: Architecture (``ModelConfig``).
        vocab_size: Joint (source = target) vocabulary size, from the tokenizer.
        pad_id: Padding token ID (embedding padding row and masks).

    Attributes:
        debug_checks: When True, validate mask invariants, token ranges and finite logits
            on every call. These checks synchronize with the GPU; use for debugging and
            the Kaggle preflight only.
    """

    output_projection: nn.Parameter | None  # None when tied (ADR-015 §3)

    def __init__(self, config: ModelConfig, *, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not 0 <= pad_id < vocab_size:
            raise ValueError(f"pad_id {pad_id} outside [0, {vocab_size})")
        self.config = config
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.debug_checks = False

        def embedding() -> TokenEmbedding:
            return TokenEmbedding(
                vocab_size, config.d_model, padding_idx=pad_id, scale=config.scale_embedding
            )

        # Registered exactly once per distinct matrix -> unique state_dict keys (ADR-015 §3).
        if config.tie_weights:
            self.embeddings = nn.ModuleDict({"shared": embedding()})
            self.register_parameter("output_projection", None)
        else:
            self.embeddings = nn.ModuleDict({"source": embedding(), "target": embedding()})
            self.output_projection = nn.Parameter(torch.empty(vocab_size, config.d_model))
        self.src_positional = _make_positional(config)
        self.tgt_positional = _make_positional(config)

        stack_kwargs = {
            "dropout": config.dropout,
            "attention_dropout": config.attention_dropout,
            "activation_dropout": config.activation_dropout,
            "norm_position": config.norm_position,
            "eps": config.layer_norm_eps,
        }
        self.encoder = Encoder(
            config.num_encoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            **stack_kwargs,  # type: ignore[arg-type]
        )
        self.decoder = Decoder(
            config.num_decoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            **stack_kwargs,  # type: ignore[arg-type]
        )
        self._reset_output_projection()

    # ------------------------------------------------------------------ #
    # Weight access
    # ------------------------------------------------------------------ #
    @property
    def src_embedding(self) -> TokenEmbedding:
        key = "shared" if self.config.tie_weights else "source"
        return cast(TokenEmbedding, self.embeddings[key])

    @property
    def tgt_embedding(self) -> TokenEmbedding:
        key = "shared" if self.config.tie_weights else "target"
        return cast(TokenEmbedding, self.embeddings[key])

    @property
    def output_weight(self) -> torch.Tensor:
        """Pre-softmax projection matrix ``[V, d]`` (the shared embedding when tied)."""
        if self.output_projection is not None:
            return self.output_projection
        return self.tgt_embedding.weight

    # ------------------------------------------------------------------ #
    # Initialization (ADR-011, ADR-015 §6)
    # ------------------------------------------------------------------ #
    def _reset_output_projection(self) -> None:
        if self.output_projection is not None:
            nn.init.normal_(self.output_projection, mean=0.0, std=self.config.d_model**-0.5)

    def reset_parameters(self) -> None:
        """Re-initialize every parameter per ADR-011, in registration order.

        Only project modules are reset; raw ``nn.Linear`` defaults are never applied.
        """
        for module in self.modules():
            if module is not self and isinstance(module, _SELF_INITIALIZING):
                module.reset_parameters()
        self._reset_output_projection()

    # ------------------------------------------------------------------ #
    # Forward path
    # ------------------------------------------------------------------ #
    def _check_tokens(self, tokens: torch.Tensor, name: str) -> None:
        if tokens.dim() != 2:
            raise ValueError(f"{name} must have shape [B, L], got {tuple(tokens.shape)}")
        if tokens.dtype not in (torch.int64, torch.int32):
            raise TypeError(f"{name} must be int64/int32 token IDs, got {tokens.dtype}")
        if tokens.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one position")
        if self.debug_checks:
            low, high = int(tokens.min()), int(tokens.max())
            if low < 0 or high >= self.vocab_size:
                raise ValueError(
                    f"{name} IDs span [{low}, {high}], vocabulary is {self.vocab_size}"
                )

    def encode(
        self, src: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """``src [B,S]`` -> ``(memory [B,S,d], src_mask [B,1,1,S], encoder maps)``."""
        self._check_tokens(src, "src")
        src_mask = encoder_attention_mask(src, self.pad_id)
        if self.debug_checks:
            assert_valid_attention_mask(src_mask, num_keys=src.shape[1])
        x = self.src_positional(self.src_embedding(src))
        memory, maps = self.encoder(x, src_mask, need_weights=need_weights)
        return memory, src_mask, maps

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """``tgt_in [B,T]`` + memory -> ``(hidden [B,T,d], self maps, cross maps)``."""
        self._check_tokens(tgt_in, "tgt_in")
        if tgt_in.shape[0] != memory.shape[0]:
            raise ValueError(
                f"batch mismatch: tgt_in {tgt_in.shape[0]} vs memory {memory.shape[0]}"
            )
        tgt_mask = decoder_attention_mask(tgt_in, self.pad_id)
        if self.debug_checks:
            assert_valid_attention_mask(
                tgt_mask, num_queries=tgt_in.shape[1], num_keys=tgt_in.shape[1]
            )
        y = self.tgt_positional(self.tgt_embedding(tgt_in))
        hidden, self_maps, cross_maps = self.decoder(
            y, memory, tgt_mask, src_mask, need_weights=need_weights
        )
        return hidden, self_maps, cross_maps

    def generator(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden [B,T,d] -> logits [B,T,V]`` = ``h Eᵀ`` (no bias, no output scaling).

        Returns raw logits; the loss (Week 5) applies ``log_softmax`` in float32 (ADR-014).
        """
        return F.linear(hidden, self.output_weight)

    def _run(self, src: torch.Tensor, tgt_in: torch.Tensor, need_weights: bool) -> ModelOutput:
        if src.shape[0] != tgt_in.shape[0]:
            raise ValueError(f"batch mismatch: src {src.shape[0]} vs tgt_in {tgt_in.shape[0]}")
        memory, src_mask, enc_maps = self.encode(src, need_weights=need_weights)
        hidden, self_maps, cross_maps = self.decode(
            tgt_in, memory, src_mask, need_weights=need_weights
        )
        logits = self.generator(hidden)
        if self.debug_checks and not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite logits")
        return ModelOutput(logits, AttentionMaps(enc_maps, self_maps, cross_maps))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits ``[B, T, V]`` for ``src [B, S]`` and ``tgt_in [B, T]``."""
        return self._run(src, tgt_in, need_weights=False).logits

    def forward_with_attention(self, src: torch.Tensor, tgt_in: torch.Tensor) -> ModelOutput:
        """Logits plus detached per-layer attention maps (visualization and analysis)."""
        return self._run(src, tgt_in, need_weights=True)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, pad_id={self.pad_id}, tied={self.config.tie_weights}"


def build_transformer(
    config: ModelConfig, *, vocab_size: int, pad_id: int, seed: int | None = None
) -> Transformer:
    """Construct a ``Transformer``; with ``seed``, initialization is a pure function of
    ``(config, vocab_size, seed)`` and leaves the global RNG state untouched."""
    if seed is None:
        return Transformer(config, vocab_size=vocab_size, pad_id=pad_id)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Transformer(config, vocab_size=vocab_size, pad_id=pad_id)


def embedding_scale(config: ModelConfig) -> float:
    """``sqrt(d)`` if embeddings are scaled (paper §3.4), else 1."""
    return math.sqrt(config.d_model) if config.scale_embedding else 1.0
