"""MultiHeadAttention vs ``torch.nn.MultiheadAttention`` with copied weights (ADR-004).

Checks outputs, per-head attention weights, input gradients and every parameter
gradient, for self-attention, causal self-attention and cross-attention with padding.
"""

from __future__ import annotations

import pytest
import torch

from aiayn.modules.masking import causal_mask, padding_mask
from aiayn.modules.multihead import MultiHeadAttention
from tests.parity.adapters import (
    TOLERANCE,
    reference_mha,
    to_torch_attn_mask,
    to_torch_key_padding_mask,
    unpack_reference_grads,
)

pytestmark = pytest.mark.parity

D, H = 32, 4


def tokens_with_padding(length: int) -> torch.Tensor:
    tokens = torch.full((3, length), 7)
    tokens[0, length - 3 :] = 0
    tokens[2, length - 1 :] = 0
    return tokens


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("case", ["self", "causal-self", "cross-padded", "batch-of-one"])
def test_parity_with_nn_multihead_attention(dtype: torch.dtype, case: str) -> None:
    torch.manual_seed(0)
    ours = MultiHeadAttention(D, H).to(dtype)
    with torch.no_grad():  # non-zero biases so their gradients are checked too
        for p in ours.parameters():
            if p.dim() == 1:
                p.normal_()
    ref = reference_mha(ours)

    tq, tk = (6, 6) if case != "cross-padded" else (5, 8)
    batch = 1 if case == "batch-of-one" else 3
    query = torch.randn(batch, tq, D, dtype=dtype)
    memory = query if case != "cross-padded" else torch.randn(batch, tk, D, dtype=dtype)

    our_mask = None
    ref_kwargs: dict[str, torch.Tensor] = {}
    if case == "causal-self":
        our_mask = causal_mask(tq)
        ref_kwargs["attn_mask"] = to_torch_attn_mask(our_mask)
    elif case == "cross-padded":
        our_mask = padding_mask(tokens_with_padding(tk), 0)
        ref_kwargs["key_padding_mask"] = to_torch_key_padding_mask(our_mask)

    q1 = query.clone().requires_grad_()
    m1 = q1 if memory is query else memory.clone().requires_grad_()
    q2 = query.clone().requires_grad_()
    m2 = q2 if memory is query else memory.clone().requires_grad_()

    out, weights = ours(q1, m1, m1, our_mask)
    ref_out, ref_weights = ref(
        q2, m2, m2, need_weights=True, average_attn_weights=False, **ref_kwargs
    )
    torch.testing.assert_close(out, ref_out, **TOLERANCE[dtype])
    assert ref_weights is not None
    torch.testing.assert_close(weights.to(dtype), ref_weights, **TOLERANCE[dtype])

    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    (ref_out * upstream).sum().backward()
    torch.testing.assert_close(q1.grad, q2.grad, **TOLERANCE[dtype])
    if memory is not query:
        torch.testing.assert_close(m1.grad, m2.grad, **TOLERANCE[dtype])
    ours_grads = {name: p.grad for name, p in ours.named_parameters()}
    for name, ref_grad in unpack_reference_grads(ref).items():
        torch.testing.assert_close(ours_grads[name], ref_grad, **TOLERANCE[dtype], msg=name)
