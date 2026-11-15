"""Add & Norm wrapper: placement of LayerNorm and residual dropout (ADR-003, ADR-012)."""

from __future__ import annotations

import pytest
import torch

from aiayn.modules.sublayer import SublayerConnection

D = 8


def x_input() -> torch.Tensor:
    return torch.randn(2, 5, D, generator=torch.Generator().manual_seed(0))


def double(t: torch.Tensor) -> torch.Tensor:
    return 2.0 * t + 1.0


def test_post_ln_formula() -> None:
    block = SublayerConnection(D, norm_position="post").eval()
    x = x_input()
    torch.testing.assert_close(block(x, double), block.norm(x + double(x)))


def test_pre_ln_formula() -> None:
    block = SublayerConnection(D, norm_position="pre").eval()
    x = x_input()
    torch.testing.assert_close(block(x, double), x + double(block.norm(x)))


def test_zero_sublayer_shows_where_the_norm_sits() -> None:
    x = x_input()
    post = SublayerConnection(D, norm_position="post")
    pre = SublayerConnection(D, norm_position="pre")
    torch.testing.assert_close(post(x, torch.zeros_like), post.norm(x))
    assert torch.equal(pre(x, torch.zeros_like), x)  # exact identity path


def test_dropout_hits_only_the_sublayer_output() -> None:
    torch.manual_seed(0)
    x = x_input()
    # Zero sub-layer output: dropout must not touch the residual input x.
    pre = SublayerConnection(D, dropout=0.9, norm_position="pre").train()
    assert torch.equal(pre(x, torch.zeros_like), x)
    # Identity sub-layer under Pre-LN: each element is x or x + LN(x)/(1-p).
    pre = SublayerConnection(D, dropout=0.5, norm_position="pre").train()
    out = pre(x, lambda h: h)
    normalized = pre.norm(x)
    kept = torch.isclose(out, x + 2 * normalized)
    dropped = torch.isclose(out, x)
    assert (kept | dropped).all() and kept.any() and dropped.any()


def test_dropout_disabled_in_eval() -> None:
    block = SublayerConnection(D, dropout=0.5).eval()
    x = x_input()
    torch.testing.assert_close(block(x, double), block(x, double))


def test_shape_changing_sublayer_is_rejected() -> None:
    block = SublayerConnection(D)
    with pytest.raises(ValueError, match="preserve shape"):
        block(x_input(), lambda h: h[..., :4])


def test_validation_and_parameters() -> None:
    with pytest.raises(ValueError):
        SublayerConnection(D, norm_position="middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SublayerConnection(D, dropout=1.5)
    assert sum(p.numel() for p in SublayerConnection(D).parameters()) == 2 * D


def test_module_conversions_work() -> None:
    """Regression: a helper named ``_apply`` once shadowed ``nn.Module._apply`` and broke
    ``.to()``/``.double()``/``.cuda()``. Every module must survive dtype/device moves."""
    from aiayn.modules import (
        LayerNorm,
        LearnedPositionalEmbedding,
        MultiHeadAttention,
        PositionwiseFeedForward,
        SinusoidalPositionalEncoding,
        TokenEmbedding,
    )

    modules = [
        SublayerConnection(D),
        LayerNorm(D),
        MultiHeadAttention(D, 2),
        PositionwiseFeedForward(D, 16),
        SinusoidalPositionalEncoding(D, 8),
        LearnedPositionalEmbedding(D, 8),
        TokenEmbedding(10, D, padding_idx=0),
    ]
    for module in modules:
        module.double().float().to("cpu").half().float()
