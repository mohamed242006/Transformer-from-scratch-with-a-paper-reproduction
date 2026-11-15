from __future__ import annotations

import pytest
import torch

from aiayn.modules.masking import causal_mask, padding_mask
from aiayn.modules.multihead import MultiHeadAttention, merge_heads, split_heads

D, H = 16, 4


@pytest.fixture
def mha() -> MultiHeadAttention:
    torch.manual_seed(0)
    return MultiHeadAttention(D, H).double().eval()


def rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64)


def test_self_and_cross_attention_shapes(mha: MultiHeadAttention) -> None:
    x, memory = rand(2, 5, D), rand(2, 9, D)
    out, w = mha(x, x, x)
    assert out.shape == (2, 5, D) and w.shape == (2, H, 5, 5)
    out, w = mha(x, memory, memory)
    assert out.shape == (2, 5, D) and w.shape == (2, H, 5, 9)


@pytest.mark.parametrize(("bias", "expected"), [(True, 4 * D * D + 4 * D), (False, 4 * D * D)])
def test_parameter_count(bias: bool, expected: int) -> None:
    module = MultiHeadAttention(D, H, bias=bias)
    assert sum(p.numel() for p in module.parameters()) == expected


def test_split_merge_round_trip_and_column_assignment() -> None:
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    heads = split_heads(x, 4)
    assert heads.shape == (2, 4, 3, 2)
    for i in range(4):  # head i holds columns [2i, 2i + 2) of every position
        assert torch.equal(heads[:, i], x[..., 2 * i : 2 * i + 2])
    assert torch.equal(merge_heads(heads), x)
    with pytest.raises(ValueError):
        split_heads(x, 3)


def test_heads_are_independent(mha: MultiHeadAttention) -> None:
    """Changing head j's query projection must not change any other head's weights."""
    x = rand(2, 6, D)
    _, before = mha(x, x, x)
    with torch.no_grad():
        mha.q_proj.weight[2 * 4 : 3 * 4] += 1.0  # rows of head 2 (head_dim = 4)
    _, after = mha(x, x, x)
    for head in range(H):
        same = torch.allclose(after[:, head], before[:, head])
        assert same == (head != 2)


def test_causal_mask_blocks_future_positions(mha: MultiHeadAttention) -> None:
    x = rand(1, 6, D)
    mask = causal_mask(6)
    base, _ = mha(x, x, x, mask)
    for t in range(1, 6):
        perturbed = x.clone()
        perturbed[:, t:] += 10.0
        out, _ = mha(perturbed, perturbed, perturbed, mask)
        torch.testing.assert_close(out[:, :t], base[:, :t])


def test_padded_keys_do_not_affect_real_outputs(mha: MultiHeadAttention) -> None:
    tokens = torch.tensor([[5, 6, 7, 0, 0]])
    mask = padding_mask(tokens, 0)
    x = rand(1, 5, D)
    base, w = mha(x, x, x, mask)
    assert torch.count_nonzero(w[..., 3:]) == 0
    perturbed = x.clone()
    perturbed[:, 3:] = 1e3
    out, _ = mha(x, perturbed, perturbed, mask)
    torch.testing.assert_close(out, base)


def test_dropout_only_in_training() -> None:
    torch.manual_seed(0)
    module = MultiHeadAttention(D, H, dropout=0.5)
    x = torch.randn(2, 5, D)
    module.eval()
    torch.testing.assert_close(module(x, x, x)[0], module(x, x, x)[0])
    module.train()
    assert not torch.allclose(module(x, x, x)[0], module(x, x, x)[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 10, "num_heads": 4},
        {"d_model": 0, "num_heads": 1},
        {"d_model": 8, "num_heads": 0},
    ],
)
def test_invalid_construction(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(**kwargs)


def test_input_validation(mha: MultiHeadAttention) -> None:
    x = rand(2, 5, D)
    with pytest.raises(ValueError):
        mha(x[0], x[0], x[0])
    with pytest.raises(ValueError):
        mha(rand(2, 5, D + 1), x, x)
    with pytest.raises(ValueError, match="batch"):
        mha(x, rand(3, 5, D), rand(3, 5, D))
    with pytest.raises(ValueError, match="lengths"):
        mha(x, rand(2, 5, D), rand(2, 4, D))
