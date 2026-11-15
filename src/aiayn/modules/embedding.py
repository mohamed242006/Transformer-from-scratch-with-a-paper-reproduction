"""Token embeddings with √d scaling (paper §3.4; math.md §5).

The weight matrix is a plain ``nn.Parameter`` so that Week 4 can tie it to the decoder
input embedding and the pre-softmax projection (three-way tying).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from aiayn.modules._validation import check_positive

__all__ = ["TokenEmbedding"]


class TokenEmbedding(nn.Module):
    """Lookup table ``E ∈ R^{V×d}`` followed by multiplication with ``sqrt(d)``.

    Shapes:
        tokens ``[..., L]`` (int64 or int32) -> output ``[..., L, d]`` (weight dtype).

    Args:
        vocab_size: Number of rows ``V``.
        d_model: Embedding width ``d``.
        padding_idx: Row whose vector is zero at initialization and receives no gradient
            (``None`` disables this). Pass the project's ``PAD_ID``.
        scale: Multiply by ``sqrt(d)`` (paper-exact). Disable only for ablations.

    Initialization (ADR-011): ``N(0, 1/d)``, so scaled outputs have unit variance and the
    same matrix used unscaled as an output projection yields logits of about unit variance.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        padding_idx: int | None,
        scale: bool = True,
    ) -> None:
        super().__init__()
        check_positive("vocab_size", vocab_size)
        check_positive("d_model", d_model)
        if padding_idx is not None and not 0 <= padding_idx < vocab_size:
            raise ValueError(f"padding_idx {padding_idx} outside [0, {vocab_size})")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self.reset_parameters()

    @property
    def scale_factor(self) -> float:
        return math.sqrt(self.d_model) if self.scale else 1.0

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.d_model**-0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dtype not in (torch.int64, torch.int32):
            raise TypeError(f"tokens must be int64 or int32, got {tokens.dtype}")
        out = F.embedding(tokens, self.weight, padding_idx=self.padding_idx)
        return out * self.scale_factor if self.scale else out

    def extra_repr(self) -> str:
        return (
            f"{self.vocab_size}, {self.d_model}, padding_idx={self.padding_idx}, scale={self.scale}"
        )
