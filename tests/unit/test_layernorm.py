"""Hand-written LayerNorm: definition and the properties derived in math.md §6."""

from __future__ import annotations

import pytest
import torch

from aiayn.modules.layernorm import LayerNorm


def rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64) * 3 + 2


def test_normalizes_each_position_to_zero_mean_unit_biased_variance() -> None:
    ln = LayerNorm(32, eps=1e-12).double()
    y = ln(rand(4, 7, 32))
    torch.testing.assert_close(
        y.mean(-1), torch.zeros(4, 7, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        y.var(-1, unbiased=False), torch.ones(4, 7, dtype=torch.float64), atol=1e-9, rtol=0
    )


def test_matches_definition_with_affine_parameters() -> None:
    ln = LayerNorm(8).double()
    with torch.no_grad():
        ln.weight.copy_(torch.linspace(0.5, 2, 8))  # type: ignore[union-attr]
        ln.bias.copy_(torch.linspace(-1, 1, 8))  # type: ignore[union-attr]
    x = rand(3, 8)
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + 1e-5) * ln.weight + ln.bias
    torch.testing.assert_close(ln(x), expected, atol=1e-12, rtol=0)


def test_invariant_to_input_scale_and_shift() -> None:
    ln = LayerNorm(16, eps=1e-12).double()
    x = rand(5, 16)
    torch.testing.assert_close(ln(3.7 * x - 11.0), ln(x), atol=1e-10, rtol=0)


def test_gradient_is_orthogonal_to_ones_and_normalized_input() -> None:
    """math.md §6.3: dL/dx is orthogonal to 1 and to x_hat (with gamma = 1)."""
    ln = LayerNorm(16, eps=1e-12).double()
    x = rand(16).requires_grad_()
    upstream = torch.randn(16, dtype=torch.float64)
    (ln(x) * upstream).sum().backward()
    assert x.grad is not None
    x_hat = (x - x.mean()) / x.std(unbiased=False)
    assert abs(x.grad.sum().item()) < 1e-10
    assert abs((x.grad * x_hat.detach()).sum().item()) < 1e-10


def test_parameters_and_reset() -> None:
    ln = LayerNorm(8)
    assert sum(p.numel() for p in ln.parameters()) == 16
    with torch.no_grad():
        ln.weight.fill_(3)  # type: ignore[union-attr]
    ln.reset_parameters()
    assert torch.equal(ln.weight, torch.ones(8)) and torch.equal(ln.bias, torch.zeros(8))
    plain = LayerNorm(8, elementwise_affine=False)
    assert list(plain.parameters()) == []
    y = plain(rand(2, 8).float())
    torch.testing.assert_close(y.mean(-1), torch.zeros(2), atol=1e-6, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_output_dtype_matches_input(dtype: torch.dtype) -> None:
    assert LayerNorm(8)(torch.randn(2, 8).to(dtype)).dtype == dtype


def test_constant_input_is_finite_and_equals_bias() -> None:
    ln = LayerNorm(8)
    y = ln(torch.full((2, 8), 5.0))
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros(2, 8))


def test_validation() -> None:
    with pytest.raises(ValueError):
        LayerNorm(0)
    with pytest.raises(ValueError):
        LayerNorm(8, eps=0.0)
    ln = LayerNorm(8)
    with pytest.raises(ValueError):
        ln(torch.randn(2, 7))
    with pytest.raises(TypeError):
        ln(torch.ones(2, 8, dtype=torch.long))
