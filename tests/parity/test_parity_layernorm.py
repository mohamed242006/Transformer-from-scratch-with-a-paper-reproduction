"""LayerNorm vs ``torch.nn.LayerNorm``, forward and backward (ADR-004)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aiayn.modules.layernorm import LayerNorm
from tests.parity.adapters import TOLERANCE

pytestmark = pytest.mark.parity


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("shape", [(2, 7, 16), (1, 1, 16), (3, 16), (5, 4, 3, 16)])
def test_forward_and_backward_parity(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    d = shape[-1]
    ours = LayerNorm(d).to(dtype)
    ref = nn.LayerNorm(d, eps=1e-5, dtype=dtype)
    with torch.no_grad():
        for module in (ours, ref):
            assert module.weight is not None and module.bias is not None
        gamma = torch.randn(d, dtype=dtype)
        beta = torch.randn(d, dtype=dtype)
        ours.weight.copy_(gamma)  # type: ignore[union-attr]
        ours.bias.copy_(beta)  # type: ignore[union-attr]
        ref.weight.copy_(gamma)
        ref.bias.copy_(beta)

    x = (torch.randn(*shape, dtype=dtype) * 4 + 1).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    upstream = torch.randn(*shape, dtype=dtype)

    y, y_ref = ours(x), ref(x_ref)
    torch.testing.assert_close(y, y_ref, **TOLERANCE[dtype])
    (y * upstream).sum().backward()
    (y_ref * upstream).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, **TOLERANCE[dtype])
    torch.testing.assert_close(ours.weight.grad, ref.weight.grad, **TOLERANCE[dtype])  # type: ignore[union-attr]
    torch.testing.assert_close(ours.bias.grad, ref.bias.grad, **TOLERANCE[dtype])  # type: ignore[union-attr]
