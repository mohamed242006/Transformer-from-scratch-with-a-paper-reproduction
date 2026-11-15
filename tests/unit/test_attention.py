"""Scaled dot-product attention against its closed form (paper Eq. 1)."""

from __future__ import annotations

import math

import pytest
import torch

from aiayn.modules.attention import scaled_dot_product_attention as sdpa
from aiayn.modules.masking import causal_mask


def closed_form(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(-1)
    return weights @ v, weights


def qkv(
    b: int = 2, h: int = 3, tq: int = 5, tk: int = 7, dk: int = 8, dv: int = 6, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(b, h, tq, dk, generator=g, dtype=dtype),
        torch.randn(b, h, tk, dk, generator=g, dtype=dtype),
        torch.randn(b, h, tk, dv, generator=g, dtype=dtype),
    )


def test_matches_closed_form_without_mask() -> None:
    q, k, v = qkv()
    out, w = sdpa(q, k, v)
    ref_out, ref_w = closed_form(q, k, v)
    assert out.shape == (2, 3, 5, 6) and w.shape == (2, 3, 5, 7)
    torch.testing.assert_close(out, ref_out, atol=1e-12, rtol=0)
    torch.testing.assert_close(w, ref_w, atol=1e-12, rtol=0)


def test_matches_closed_form_with_padding_mask_and_masked_weights_are_exactly_zero() -> None:
    q, k, v = qkv()
    mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    mask[0, ..., 4:] = False
    out, w = sdpa(q, k, v, mask)
    ref_out, _ = closed_form(q, k, v, mask)
    torch.testing.assert_close(out, ref_out, atol=1e-12, rtol=0)
    assert torch.count_nonzero(w[0, ..., 4:]) == 0
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 3, 5, dtype=torch.float64))


def test_causal_mask_with_query_specific_rows() -> None:
    q, k, v = qkv(tq=6, tk=6)
    mask = causal_mask(6)
    out, w = sdpa(q, k, v, mask)
    assert torch.count_nonzero(torch.triu(w, diagonal=1)) == 0
    torch.testing.assert_close(out, closed_form(q, k, v, mask)[0], atol=1e-12, rtol=0)
    torch.testing.assert_close(w[..., 0, 0], torch.ones(2, 3, dtype=torch.float64))


def test_works_without_head_dimension() -> None:
    q, k, v = (t[:, 0] for t in qkv())  # [B, T, d]
    out, w = sdpa(q, k, v)
    assert out.shape == (2, 5, 6) and w.shape == (2, 5, 7)


def test_weights_are_computed_in_at_least_float32() -> None:
    q, k, v = qkv(dtype=torch.float32)
    out, w = sdpa(q, k, v)
    assert out.dtype == torch.float32 and w.dtype == torch.float32
    q, k, v = qkv(dtype=torch.float64)
    assert sdpa(q, k, v)[1].dtype == torch.float64


def test_zero_queries_give_uniform_attention() -> None:
    _, k, v = qkv()
    q = torch.zeros(2, 3, 5, 8, dtype=torch.float64)
    out, w = sdpa(q, k, v)
    torch.testing.assert_close(w, torch.full_like(w, 1 / 7))
    torch.testing.assert_close(out, v.mean(-2, keepdim=True).expand_as(out))


def test_dropout_applies_only_in_training_and_returned_weights_are_pre_dropout() -> None:
    q, k, v = qkv(dtype=torch.float32)
    eval_out, eval_w = sdpa(q, k, v, dropout_p=0.5, training=False)
    torch.testing.assert_close(eval_out, sdpa(q, k, v)[0])
    torch.manual_seed(0)
    train_out, train_w = sdpa(q, k, v, dropout_p=0.5, training=True)
    assert not torch.allclose(train_out, eval_out)
    torch.testing.assert_close(train_w, eval_w)  # probabilities, not the dropped ones


@pytest.mark.parametrize("p", [-0.1, 1.0])
def test_invalid_dropout(p: float) -> None:
    q, k, v = qkv()
    with pytest.raises(ValueError):
        sdpa(q, k, v, dropout_p=p)


def test_input_validation() -> None:
    q, k, v = qkv()
    with pytest.raises(TypeError, match="dtype mismatch"):
        sdpa(q.float(), k, v)
    with pytest.raises(TypeError, match="floating"):
        sdpa(q.long(), k.long(), v.long())
    with pytest.raises(ValueError, match="feature sizes"):
        sdpa(q[..., :4], k, v)
    with pytest.raises(ValueError, match="lengths differ"):
        sdpa(q, k, v[..., :3, :])
    with pytest.raises(ValueError, match="leading"):
        sdpa(q[:1], k, v)
    with pytest.raises(ValueError):
        sdpa(q[0, 0, 0], k[0, 0, 0], v[0, 0, 0])


def test_mask_validation() -> None:
    q, k, v = qkv()
    with pytest.raises(TypeError, match="bool"):
        sdpa(q, k, v, torch.ones(2, 1, 1, 7))
    with pytest.raises(ValueError, match="rank"):
        sdpa(q, k, v, torch.ones(2, 7, dtype=torch.bool))
    with pytest.raises(ValueError, match="broadcastable"):
        sdpa(q, k, v, torch.ones(2, 1, 1, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="broadcastable"):
        sdpa(q, k, v, torch.ones(2, 1, 3, 7, dtype=torch.bool))
