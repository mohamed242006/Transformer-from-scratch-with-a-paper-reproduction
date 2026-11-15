"""Scaled dot-product attention (paper §3.2.1, Eq. 1; math.md §1–2).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Precision policy (ADR-014): the QK^T matmul runs in the input dtype; scores are upcast to
at least float32 before masking and softmax; masked logits are set to the dtype's most
negative finite value with ``masked_fill`` (never -inf, never -1e9), so masked weights are
exactly zero and fp16 cannot overflow.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from aiayn.modules._validation import check_dropout, check_floating

__all__ = ["scaled_dot_product_attention"]


def _validate(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None
) -> None:
    for name, t in (("query", query), ("key", key), ("value", value)):
        check_floating(t, name)
        if t.dim() < 2:
            raise ValueError(f"{name} must have shape [..., T, d], got {tuple(t.shape)}")
    if not (query.dtype == key.dtype == value.dtype):
        raise TypeError(f"dtype mismatch: q={query.dtype}, k={key.dtype}, v={value.dtype}")
    if not (query.shape[:-2] == key.shape[:-2] == value.shape[:-2]):
        raise ValueError(
            "leading (batch/head) dimensions differ: "
            f"q={tuple(query.shape)}, k={tuple(key.shape)}, v={tuple(value.shape)}"
        )
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query/key feature sizes differ: {query.shape[-1]} vs {key.shape[-1]}")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(f"key/value lengths differ: {key.shape[-2]} vs {value.shape[-2]}")
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be torch.bool (True = may attend), got {mask.dtype}")
        if mask.dim() != query.dim():
            raise ValueError(
                f"mask rank {mask.dim()} != query rank {query.dim()}; "
                "build masks with aiayn.modules.masking (ADR-008)"
            )
        if mask.shape[-1] != key.shape[-2] or mask.shape[-2] not in (1, query.shape[-2]):
            raise ValueError(
                f"mask {tuple(mask.shape)} not broadcastable to scores "
                f"[..., {query.shape[-2]}, {key.shape[-2]}]"
            )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention outputs and weights.

    Shapes (any leading dims ``...``, typically ``[B, h]``):
        query ``[..., Tq, d_k]``, key ``[..., Tk, d_k]``, value ``[..., Tk, d_v]``,
        mask ``[..., Tq | 1, Tk]`` bool, broadcastable, True = may attend (ADR-008).

    Returns:
        ``(output [..., Tq, d_v]`` in the value dtype,
        ``weights [..., Tq, Tk]`` in the compute dtype (≥ float32)), where ``weights``
        are the **pre-dropout** probabilities (rows sum to 1; masked entries are exactly 0).

    Args:
        dropout_p: Attention dropout probability (ADR-012), applied only if ``training``.

    Note:
        A fully masked query row would yield a *uniform* distribution here (all logits
        equal), not NaN. ADR-008 framing guarantees this never happens; use
        ``masking.assert_valid_attention_mask`` in debug runs to verify.
    """
    _validate(query, key, value, mask)
    check_dropout("dropout_p", dropout_p)

    d_k = query.shape[-1]
    scores = torch.matmul(query * (1.0 / math.sqrt(d_k)), key.transpose(-2, -1))  # [..., Tq, Tk]
    compute_dtype = torch.promote_types(scores.dtype, torch.float32)
    scores = scores.to(compute_dtype)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(compute_dtype).min)
    weights = torch.softmax(scores, dim=-1)
    attended = F.dropout(weights, p=dropout_p, training=training) if dropout_p > 0 else weights
    output = torch.matmul(attended.to(value.dtype), value)  # [..., Tq, d_v]
    return output, weights
