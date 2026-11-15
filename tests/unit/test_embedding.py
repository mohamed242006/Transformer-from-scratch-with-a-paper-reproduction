from __future__ import annotations

import math

import pytest
import torch

from aiayn.modules.embedding import TokenEmbedding

PAD = 0


def test_lookup_and_sqrt_d_scaling() -> None:
    emb = TokenEmbedding(50, 16, padding_idx=PAD)
    tokens = torch.tensor([[1, 2, 3], [4, PAD, PAD]])
    out = emb(tokens)
    assert out.shape == (2, 3, 16)
    torch.testing.assert_close(out, emb.weight[tokens] * math.sqrt(16))
    assert torch.equal(emb(tokens.int()), out)  # int32 accepted


def test_scale_can_be_disabled_for_ablation() -> None:
    emb = TokenEmbedding(10, 4, padding_idx=None, scale=False)
    tokens = torch.tensor([[1, 2]])
    torch.testing.assert_close(emb(tokens), emb.weight[tokens])


def test_initialization_gives_unit_variance_after_scaling() -> None:
    torch.manual_seed(0)
    emb = TokenEmbedding(4000, 64, padding_idx=PAD)
    std = emb.weight[1:].std().item()
    assert abs(std - 64**-0.5) / 64**-0.5 < 0.03
    scaled = emb(torch.arange(1, 4000)[None])
    assert abs(scaled.var().item() - 1.0) < 0.05


def test_padding_row_is_zero_and_receives_no_gradient() -> None:
    emb = TokenEmbedding(10, 4, padding_idx=PAD)
    assert torch.count_nonzero(emb.weight[PAD]) == 0
    emb(torch.tensor([[PAD, 3, PAD]])).sum().backward()
    assert emb.weight.grad is not None
    assert torch.count_nonzero(emb.weight.grad[PAD]) == 0
    assert torch.count_nonzero(emb.weight.grad[3]) == 4


def test_weight_is_a_single_tieable_parameter() -> None:
    emb = TokenEmbedding(10, 4, padding_idx=PAD)
    params = list(emb.parameters())
    assert len(params) == 1 and params[0] is emb.weight and params[0].numel() == 40


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0, "d_model": 4, "padding_idx": None},
        {"vocab_size": 4, "d_model": 0, "padding_idx": None},
        {"vocab_size": 4, "d_model": 4, "padding_idx": 4},
        {"vocab_size": 4, "d_model": 4, "padding_idx": -1},
    ],
)
def test_invalid_construction(kwargs: dict[str, int | None]) -> None:
    with pytest.raises(ValueError):
        TokenEmbedding(**kwargs)  # type: ignore[arg-type]


def test_rejects_float_tokens_and_out_of_range_ids() -> None:
    emb = TokenEmbedding(10, 4, padding_idx=PAD)
    with pytest.raises(TypeError):
        emb(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(IndexError):
        emb(torch.tensor([[10]]))
