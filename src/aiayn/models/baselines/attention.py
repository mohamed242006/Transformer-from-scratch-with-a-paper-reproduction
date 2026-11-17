"""Hand-written attention for the RNN baselines (ADR-001 allows the recurrent cells only).

Shapes: queries ``[B, T, H]``, encoder outputs ("values") ``[B, S, He]``, key mask
``[B, S]`` (True = real token). Keys are projected once per source sentence
(``project_keys``) and reused at every decoder step.

* Bahdanau et al. (2015), additive: ``e_ts = vᵀ tanh(W_q q_t + U_k h_s + b)``.
* Luong et al. (2015), global "general": ``e_ts = q_tᵀ W_a h_s``.

Softmax runs in at least float32 (ADR-014); masked keys get exactly zero weight.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["BahdanauAttention", "LuongAttention", "attend"]


def attend(
    scores: torch.Tensor, values: torch.Tensor, key_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scores [B,T,S]`` -> ``(context [B,T,He], weights [B,T,S])``.

    Every sentence has at least one real key (EOS), so no row is fully masked.
    """
    compute = torch.promote_types(scores.dtype, torch.float32)
    scores = scores.to(compute).masked_fill(~key_mask[:, None, :], float("-inf"))
    weights = F.softmax(scores, dim=-1)
    context = torch.matmul(weights.to(values.dtype), values)
    return context, weights


class LuongAttention(nn.Module):
    """Global attention with the "general" score ``qᵀ W_a h`` (Luong et al., 2015 §3.1)."""

    def __init__(self, query_dim: int, value_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(value_dim, query_dim, bias=False)  # W_a, applied to keys

    def project_keys(self, values: torch.Tensor) -> torch.Tensor:
        keys: torch.Tensor = self.key_proj(values)  # [B, S, H]
        return keys

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, keys.transpose(1, 2))  # [B, T, S]
        return attend(scores, values, key_mask)


class BahdanauAttention(nn.Module):
    """Additive attention ``vᵀ tanh(W_q q + U_k h + b)`` (Bahdanau et al., 2015, App. A)."""

    def __init__(self, query_dim: int, value_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, attention_dim, bias=False)  # W_q
        self.key_proj = nn.Linear(value_dim, attention_dim)  # U_k and b
        self.score = nn.Linear(attention_dim, 1, bias=False)  # v

    def project_keys(self, values: torch.Tensor) -> torch.Tensor:
        keys: torch.Tensor = self.key_proj(values)  # [B, S, A]
        return keys

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(query)  # [B, T, A]
        energy = torch.tanh(q[:, :, None, :] + keys[:, None, :, :])  # [B, T, S, A]
        scores = self.score(energy).squeeze(-1)  # [B, T, S]
        return attend(scores, values, key_mask)
