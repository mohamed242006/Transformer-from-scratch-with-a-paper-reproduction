"""ADR-011 initialization rules."""

from __future__ import annotations

import math

import torch

from aiayn.modules import (
    LayerNorm,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    TokenEmbedding,
)


def xavier_bound(fan_in: int, fan_out: int) -> float:
    return math.sqrt(6.0 / (fan_in + fan_out))


def test_attention_projections_are_xavier_with_zero_bias() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(256, 8)
    bound = xavier_bound(256, 256)
    for layer in (mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj):
        assert layer.weight.abs().max().item() <= bound
        assert abs(layer.weight.std().item() - bound / math.sqrt(3)) < 0.05 * bound
        assert layer.bias is not None and torch.count_nonzero(layer.bias) == 0


def test_ffn_is_xavier_with_zero_bias() -> None:
    torch.manual_seed(0)
    ffn = PositionwiseFeedForward(128, 512)
    for layer, (fan_in, fan_out) in ((ffn.linear1, (128, 512)), (ffn.linear2, (512, 128))):
        bound = xavier_bound(fan_in, fan_out)
        assert layer.weight.abs().max().item() <= bound
        assert layer.bias is not None and torch.count_nonzero(layer.bias) == 0


def test_embeddings_are_normal_with_std_inverse_sqrt_d() -> None:
    torch.manual_seed(0)
    emb = TokenEmbedding(5000, 128, padding_idx=0)
    assert abs(emb.weight[1:].std().item() - 128**-0.5) < 0.03 * 128**-0.5
    assert abs(emb.weight[1:].mean().item()) < 1e-3
    pos = LearnedPositionalEmbedding(128, 1024)
    assert abs(pos.weight.std().item() - 128**-0.5) < 0.03 * 128**-0.5


def test_layernorm_starts_as_identity_affine() -> None:
    ln = LayerNorm(16)
    assert torch.equal(ln.weight, torch.ones(16)) and torch.equal(ln.bias, torch.zeros(16))


def test_initialization_is_seed_deterministic() -> None:
    torch.manual_seed(3)
    a = MultiHeadAttention(32, 4)
    torch.manual_seed(3)
    b = MultiHeadAttention(32, 4)
    torch.manual_seed(9)
    a.reset_parameters()
    torch.manual_seed(9)
    b.reset_parameters()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)
