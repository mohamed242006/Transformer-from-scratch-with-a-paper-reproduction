"""Positional encodings: closed form and the properties derived in math.md §4."""

from __future__ import annotations

import math

import pytest
import torch

from aiayn.modules.positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
    sinusoidal_table,
)

D = 16


def table64(max_len: int = 64, d: int = D) -> torch.Tensor:
    return sinusoidal_table(max_len, d, dtype=torch.float64)


def frequencies(d: int = D) -> torch.Tensor:
    return torch.tensor([10000.0 ** (-2 * i / d) for i in range(d // 2)], dtype=torch.float64)


def test_matches_paper_formula_elementwise() -> None:
    pe = table64(20, 8)
    for pos in range(20):
        for i in range(4):
            angle = pos / 10000 ** (2 * i / 8)
            assert pe[pos, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-12)
            assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-12)


def test_position_zero_and_default_dtype() -> None:
    pe = sinusoidal_table(4, D)
    assert pe.dtype == torch.float32
    assert torch.equal(pe[0, 0::2], torch.zeros(D // 2))
    assert torch.equal(pe[0, 1::2], torch.ones(D // 2))


def test_relative_offset_is_a_rotation() -> None:
    """PE(p + k) = R(k) PE(p) with a block-diagonal rotation independent of p (math.md §4.3)."""
    pe, w = table64(), frequencies()
    for k in (1, 5, 17):
        rotation = torch.zeros(D, D, dtype=torch.float64)
        for i, wi in enumerate(w):
            c, s = math.cos(wi * k), math.sin(wi * k)
            rotation[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = torch.tensor(
                [[c, s], [-s, c]], dtype=torch.float64
            )
        for p in (0, 3, 30):
            torch.testing.assert_close(pe[p + k], rotation @ pe[p], atol=1e-10, rtol=0)


def test_dot_product_depends_only_on_distance() -> None:
    pe, w = table64(), frequencies()
    for p, q in ((3, 10), (20, 13), (7, 7)):
        expected = torch.cos(w * (p - q)).sum()
        torch.testing.assert_close(pe[p] @ pe[q], expected, atol=1e-10, rtol=0)
    norms = (pe * pe).sum(dim=-1)
    torch.testing.assert_close(norms, torch.full_like(norms, D / 2), atol=1e-10, rtol=0)


@pytest.mark.parametrize(("max_len", "d"), [(4, 3), (0, 4), (4, 0)])
def test_invalid_table_sizes(max_len: int, d: int) -> None:
    with pytest.raises(ValueError):
        sinusoidal_table(max_len, d)


def test_sinusoidal_module_adds_table_with_offset() -> None:
    module = SinusoidalPositionalEncoding(D, max_len=32).eval()
    x = torch.randn(2, 5, D)
    torch.testing.assert_close(module(x), x + module.table[:5].to(x.dtype))
    torch.testing.assert_close(module(x, offset=7), x + module.table[7:12].to(x.dtype))


def test_sinusoidal_buffer_is_not_a_parameter_nor_persisted() -> None:
    module = SinusoidalPositionalEncoding(D, max_len=32)
    assert list(module.parameters()) == []
    assert "table" not in module.state_dict()
    assert "table" in dict(module.named_buffers())


def test_output_follows_input_dtype() -> None:
    module = SinusoidalPositionalEncoding(D, max_len=8).eval()
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        assert module(torch.zeros(1, 3, D, dtype=dtype)).dtype == dtype


def test_length_beyond_max_len_is_rejected_with_guidance() -> None:
    module = SinusoidalPositionalEncoding(D, max_len=8)
    with pytest.raises(ValueError, match="max_len"):
        module(torch.zeros(1, 9, D))
    with pytest.raises(ValueError, match="max_len"):
        module(torch.zeros(1, 4, D), offset=5)
    with pytest.raises(ValueError):
        module(torch.zeros(1, 4, D), offset=-1)


def test_shape_validation() -> None:
    module = SinusoidalPositionalEncoding(D, max_len=8)
    with pytest.raises(ValueError):
        module(torch.zeros(3, D))
    with pytest.raises(ValueError):
        module(torch.zeros(1, 3, D + 2))


def test_dropout_only_in_training() -> None:
    torch.manual_seed(0)
    module = SinusoidalPositionalEncoding(D, max_len=8, dropout=0.5)
    x = torch.randn(4, 6, D)
    module.eval()
    torch.testing.assert_close(module(x), x + module.table[:6].to(x.dtype))
    module.train()
    out = module(x)
    assert (out == 0).any()  # dropped entries
    kept = out != 0
    torch.testing.assert_close(out[kept], ((x + module.table[:6].to(x.dtype)) * 2)[kept])  # 1/(1-p)


def test_learned_embedding() -> None:
    torch.manual_seed(0)
    module = LearnedPositionalEmbedding(64, max_len=512).eval()
    assert sum(p.numel() for p in module.parameters()) == 512 * 64
    assert "weight" in module.state_dict()
    assert abs(module.weight.std().item() - 64**-0.5) < 0.01
    x = torch.randn(2, 5, 64)
    torch.testing.assert_close(module(x, offset=3), x + module.weight[3:8])
    with pytest.raises(ValueError):
        module(torch.zeros(1, 513, 64))


def test_table_is_stored_in_float64_so_double_models_are_exact() -> None:
    """Regression (Week 4): a float32 table cast to float64 by model.double() carried
    ~1e-7 error into every float64 parity/gradcheck computation."""
    module = SinusoidalPositionalEncoding(D, max_len=300).double()
    assert module.table.dtype == torch.float64
    torch.testing.assert_close(module.table, table64(300), atol=0, rtol=0)
