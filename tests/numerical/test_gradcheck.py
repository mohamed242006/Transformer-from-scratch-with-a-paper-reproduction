"""Finite-difference gradient checks in float64 (spec §10.4)."""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from aiayn.modules import (
    LayerNorm,
    MultiHeadAttention,
    PositionwiseFeedForward,
    SublayerConnection,
    scaled_dot_product_attention,
)
from aiayn.modules.masking import causal_mask

pytestmark = pytest.mark.numerical


def leaf(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64, requires_grad=True)


def test_attention_inputs() -> None:
    torch.manual_seed(0)
    mask = causal_mask(4) & torch.tensor([True, True, True, False])[None, None, None, :]
    assert gradcheck(
        lambda q, k, v: scaled_dot_product_attention(q, k, v, mask)[0],
        (leaf(2, 2, 4, 3), leaf(2, 2, 4, 3), leaf(2, 2, 4, 5)),
    )


def test_layernorm_input_and_parameters() -> None:
    torch.manual_seed(0)
    ln = LayerNorm(6).double()
    params = {"weight": leaf(6), "bias": leaf(6)}
    assert gradcheck(
        lambda x, w, b: functional_call(ln, {"weight": w, "bias": b}, (x,)),
        (leaf(3, 6), params["weight"], params["bias"]),
    )


def test_multihead_attention_inputs() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(8, 2).double()
    assert gradcheck(lambda q, kv: mha(q, kv, kv)[0], (leaf(2, 3, 8), leaf(2, 4, 8)))


def test_feedforward_input() -> None:
    torch.manual_seed(0)
    ffn = PositionwiseFeedForward(4, 8).double()
    x = leaf(2, 3, 4)
    with torch.no_grad():  # keep inputs away from ReLU kinks, where gradients are undefined
        pre = ffn.linear1(x)
        ffn.linear1.bias += torch.where(pre.abs().amin(dim=(0, 1)) < 1e-3, 0.1, 0.0)
    assert gradcheck(ffn, (x,))


@pytest.mark.parametrize("position", ["post", "pre"])
def test_sublayer_connection(position: str) -> None:
    torch.manual_seed(0)
    block = SublayerConnection(6, norm_position=position).double()  # type: ignore[arg-type]
    weight = torch.randn(6, 6, dtype=torch.float64)
    assert gradcheck(lambda x: block(x, lambda h: torch.tanh(h @ weight)), (leaf(2, 3, 6),))
