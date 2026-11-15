"""Low-precision and extreme-value stability (ADR-014), emulated on CPU."""

from __future__ import annotations

import pytest
import torch

from aiayn.modules import LayerNorm, MultiHeadAttention, scaled_dot_product_attention
from aiayn.modules.masking import causal_mask

pytestmark = pytest.mark.numerical

LOW = [torch.float16, torch.bfloat16]
LOOSE = {torch.float16: {"atol": 2e-2, "rtol": 2e-2}, torch.bfloat16: {"atol": 6e-2, "rtol": 6e-2}}


def qkv(dtype: torch.dtype, scale: float = 1.0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [torch.randn(2, 4, 6, 16, generator=g) * scale for _ in range(3)]


@pytest.mark.parametrize("dtype", LOW)
def test_attention_low_precision_matches_fp32_and_masks_exactly(dtype: torch.dtype) -> None:
    q, k, v = qkv(dtype)
    mask = causal_mask(6)
    ref_out, ref_w = scaled_dot_product_attention(q, k, v, mask)
    out, w = scaled_dot_product_attention(q.to(dtype), k.to(dtype), v.to(dtype), mask)
    assert out.dtype == dtype and w.dtype == torch.float32  # softmax upcast (ADR-014)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(torch.triu(w, diagonal=1)) == 0
    torch.testing.assert_close(out.float(), ref_out, **LOOSE[dtype])
    torch.testing.assert_close(w, ref_w, **LOOSE[dtype])


def test_fp16_scores_near_the_dtype_limit_stay_finite() -> None:
    # Logits around ±4e3 after scaling: exp() would overflow any fp16 softmax.
    q, k, v = (t.half() for t in qkv(torch.float16, scale=45.0))
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    assert torch.isfinite(out).all() and torch.isfinite(w).all()
    torch.testing.assert_close(w.sum(-1), torch.ones_like(w.sum(-1)))


def test_extreme_float32_logits_are_stable() -> None:
    q, k, v = qkv(torch.float32, scale=1e4)
    out, w = scaled_dot_product_attention(q, k, v)
    assert torch.isfinite(out).all() and torch.isfinite(w).all()


def test_fully_masked_row_is_finite_and_uniform() -> None:
    """Documented behaviour (ADR-008 forbids such rows; this proves no NaN can arise)."""
    q, k, v = qkv(torch.float32)
    mask = torch.zeros(1, 1, 1, 6, dtype=torch.bool)
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(w, torch.full_like(w, 1 / 6))


@pytest.mark.parametrize("dtype", LOW)
def test_layernorm_survives_values_whose_squares_overflow_fp16(dtype: torch.dtype) -> None:
    x = torch.zeros(4, 64)
    x[:, ::2] = 600.0  # centered values ±300 -> squares 9e4 > fp16 max 65504
    x += torch.randn(4, 64)
    ln = LayerNorm(64)
    expected = ln(x)
    out = ln(x.to(dtype))
    assert out.dtype == dtype and torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, **LOOSE[dtype])

    if dtype is torch.float16:  # a naive fp16 implementation overflows on the same input
        xh = x.half()
        naive_var = ((xh - xh.mean(-1, keepdim=True)) ** 2).mean(-1)
        assert not torch.isfinite(naive_var).all()


def test_multihead_attention_under_bf16_autocast() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(32, 4).eval()
    x = torch.randn(2, 7, 32)
    mask = causal_mask(7)
    ref, _ = mha(x, x, x, mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, w = mha(x, x, x, mask)
    assert torch.isfinite(out).all() and w.dtype == torch.float32
    torch.testing.assert_close(out.float(), ref, **LOOSE[torch.bfloat16])
