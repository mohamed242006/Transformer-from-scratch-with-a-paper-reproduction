"""Scaled dot-product attention vs ``F.scaled_dot_product_attention`` (ADR-004).

``F.scaled_dot_product_attention`` uses the same boolean convention as ours (True = takes
part), so masks pass through unchanged.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aiayn.modules.attention import scaled_dot_product_attention
from aiayn.modules.masking import causal_mask
from tests.parity.adapters import TOLERANCE

pytestmark = pytest.mark.parity


def make_inputs(dtype: torch.dtype, tq: int, tk: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    shapes = [(2, 4, tq, 8), (2, 4, tk, 8), (2, 4, tk, 8)]
    return [torch.randn(*s, generator=g, dtype=dtype).requires_grad_() for s in shapes]


def padding(tk: int) -> torch.Tensor:
    mask = torch.ones(2, 1, 1, tk, dtype=torch.bool)
    mask[0, ..., tk - 2 :] = False
    return mask


CASES = {
    "no-mask": (6, 6, lambda: None),
    "padding": (6, 9, lambda: padding(9)),  # cross-attention shape
    "causal": (7, 7, lambda: causal_mask(7)),
    "causal+padding": (7, 7, lambda: causal_mask(7) & padding(7)),
}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("case", list(CASES))
def test_forward_and_backward_parity(dtype: torch.dtype, case: str) -> None:
    tq, tk, make_mask = CASES[case]
    mask = make_mask()
    ours_in = make_inputs(dtype, tq, tk)
    ref_in = [t.detach().clone().requires_grad_() for t in ours_in]

    out, _ = scaled_dot_product_attention(*ours_in, mask)
    ref = F.scaled_dot_product_attention(*ref_in, attn_mask=mask)
    torch.testing.assert_close(out, ref, **TOLERANCE[dtype])

    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    (ref * upstream).sum().backward()
    for ours_t, ref_t in zip(ours_in, ref_in, strict=True):
        torch.testing.assert_close(ours_t.grad, ref_t.grad, **TOLERANCE[dtype])
