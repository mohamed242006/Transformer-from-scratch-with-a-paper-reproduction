"""Attention masks (ADR-008).

Convention: ``torch.bool``, **True = query may attend to key**, rank 4, broadcastable to
attention scores of shape ``[B, h, Tq, Tk]``. Only keys are masked: padded *query* rows
attend normally and their outputs are ignored by the loss.

    encoder_attention_mask(src)        -> [B, 1, 1, S]   encoder self-attn + cross-attn
    causal_mask(T)                     -> [1, 1, T, T]   lower triangle incl. diagonal
    decoder_attention_mask(tgt_in)     -> [B, 1, T, T]   causal AND target key padding
"""

from __future__ import annotations

import torch

__all__ = [
    "assert_valid_attention_mask",
    "causal_mask",
    "decoder_attention_mask",
    "encoder_attention_mask",
    "padding_mask",
]


def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Key-padding mask: ``[B, L]`` token IDs -> ``[B, 1, 1, L]`` (True = real token)."""
    if tokens.dim() != 2:
        raise ValueError(f"expected token IDs of shape [B, L], got {tuple(tokens.shape)}")
    return (tokens != pad_id)[:, None, None, :]


def causal_mask(size: int, device: torch.device | str | None = None) -> torch.Tensor:
    """``[1, 1, T, T]`` with True where key position j <= query position i."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    positions = torch.arange(size, device=device)
    return (positions[None, :] <= positions[:, None])[None, None, :, :]


def encoder_attention_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mask over source keys, used by encoder self-attention and decoder cross-attention.

    ``src [B, S]`` -> ``[B, 1, 1, S]``; broadcasts over heads and over any number of
    queries (S for encoder self-attention, T for cross-attention).
    """
    return padding_mask(src, pad_id)


def decoder_attention_mask(tgt_in: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Decoder self-attention mask: ``tgt_in [B, T]`` -> ``[B, 1, T, T]`` (causal ∧ padding)."""
    keys = padding_mask(tgt_in, pad_id)  # [B, 1, 1, T]
    return keys & causal_mask(tgt_in.shape[1], device=tgt_in.device)  # [B, 1, T, T]


def assert_valid_attention_mask(
    mask: torch.Tensor, *, num_queries: int | None = None, num_keys: int | None = None
) -> None:
    """Check ADR-008 invariants; raise ``ValueError`` describing the first violation.

    Invariants: bool dtype, rank 4, head dimension 1, optional size checks and **at least
    one attendable key in every query row** (otherwise softmax yields NaN).
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be torch.bool, got {mask.dtype}")
    if mask.dim() != 4:
        raise ValueError(f"mask must have rank 4 [B, 1, Tq|1, Tk], got {tuple(mask.shape)}")
    if mask.shape[1] != 1:
        raise ValueError(f"mask head dimension must be 1 (broadcast), got {mask.shape[1]}")
    if num_queries is not None and mask.shape[2] not in (1, num_queries):
        raise ValueError(f"mask query dimension {mask.shape[2]} incompatible with {num_queries}")
    if num_keys is not None and mask.shape[3] != num_keys:
        raise ValueError(f"mask key dimension {mask.shape[3]} != {num_keys}")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("mask has a query row with no attendable key (softmax would be NaN)")
