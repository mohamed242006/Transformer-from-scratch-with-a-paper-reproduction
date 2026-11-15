"""Layer normalization, implemented by hand (Ba et al., 2016; math.md §6; ADR-001).

    LN(x) = γ ⊙ (x − μ) / sqrt(σ² + ε) + β,   μ, σ² over the last (feature) dimension

σ² is the **biased** variance (divide by d), as in the original definition and
``torch.nn.LayerNorm``. Statistics are computed in at least float32 with the two-pass
(centered) formula, which avoids the catastrophic cancellation of E[x²] − E[x]² and fp16
overflow of x² (ADR-014). The output has the input's dtype.
"""

from __future__ import annotations

import torch
from torch import nn

from aiayn.modules._validation import check_floating, check_last_dim, check_positive

__all__ = ["LayerNorm"]


class LayerNorm(nn.Module):
    """``[..., d] -> [..., d]``, normalizing each position independently.

    Args:
        d_model: Feature size ``d``.
        eps: Added to the variance inside the square root. Default ``1e-5`` matches
            ``nn.LayerNorm`` (ADR-004 parity; paper_notes U14).
        elementwise_affine: Learn ``γ`` (init 1) and ``β`` (init 0); parameters ``2d``.
    """

    def __init__(self, d_model: int, *, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        check_positive("d_model", d_model)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.d_model = d_model
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight: nn.Parameter | None
        self.bias: nn.Parameter | None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.weight is not None and self.bias is not None:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_floating(x, "x")
        check_last_dim(x, self.d_model, "x")
        compute_dtype = torch.promote_types(x.dtype, torch.float32)
        xc = x.to(compute_dtype)
        mean = xc.mean(dim=-1, keepdim=True)
        centered = xc - mean
        variance = centered.square().mean(dim=-1, keepdim=True)  # biased: / d
        normalized = centered * torch.rsqrt(variance + self.eps)
        if self.weight is not None and self.bias is not None:
            normalized = normalized * self.weight.to(compute_dtype) + self.bias.to(compute_dtype)
        return normalized.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.d_model}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
