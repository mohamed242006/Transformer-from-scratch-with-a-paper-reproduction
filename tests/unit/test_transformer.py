"""Full-model behaviour: shapes, tying, masking semantics, attention maps, initialization."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aiayn.models import AttentionMaps, Transformer, build_transformer
from aiayn.modules import LayerNorm, MultiHeadAttention
from tests.conftest import small_model_config

V, PAD = 50, 0
SRC = torch.tensor([[5, 6, 7, 8, 3], [9, 10, 3, PAD, PAD]])
TGT = torch.tensor([[2, 11, 12, 13], [2, 14, 15, PAD]])


def model(**overrides: object) -> Transformer:
    seed = int(overrides.pop("seed", 0))  # type: ignore[call-overload]
    return (
        build_transformer(small_model_config(**overrides), vocab_size=V, pad_id=PAD, seed=seed)
        .double()
        .eval()
    )


def test_forward_shapes_and_dtype() -> None:
    logits = model()(SRC, TGT)
    assert logits.shape == (2, 4, V) and logits.dtype == torch.float64


def test_three_way_weight_tying() -> None:
    m = model()
    assert m.src_embedding is m.tgt_embedding
    assert m.output_weight is m.src_embedding.weight
    hidden = torch.randn(2, 4, 32, dtype=torch.float64)
    torch.testing.assert_close(m.generator(hidden), hidden @ m.src_embedding.weight.T)
    names = [n for n, _ in m.named_parameters()]
    assert names.count("embeddings.shared.weight") == 1 and "output_projection" not in names


def test_tying_survives_parameter_updates() -> None:
    m = model()
    with torch.no_grad():
        m.src_embedding.weight[7] += 1.0
    assert torch.equal(m.output_weight[7], m.tgt_embedding.weight[7])


def test_untied_variant_has_three_independent_matrices() -> None:
    m = model(tie_weights=False)
    w = [m.src_embedding.weight, m.tgt_embedding.weight, m.output_weight]
    assert len({id(t) for t in w}) == 3
    assert not torch.equal(w[0], w[1])


def test_state_dict_has_no_aliased_tensors() -> None:
    state = model().state_dict()
    pointers = [t.data_ptr() for t in state.values()]
    assert len(pointers) == len(set(pointers))
    assert not any("positional.table" in k for k in state)  # sinusoids are not persisted


def test_decoder_is_causal_end_to_end() -> None:
    m = model()
    tgt = torch.tensor([[2, 11, 12, 13, 14]])
    src = SRC[:1]
    base = m(src, tgt)
    for t in range(1, 5):
        changed = tgt.clone()
        changed[0, t:] = 40
        out = m(src, changed)
        torch.testing.assert_close(out[:, :t], base[:, :t], atol=1e-12, rtol=0)
        assert not torch.allclose(out[:, t], base[:, t])


def test_extra_padding_does_not_change_real_outputs() -> None:
    m = model()
    base = m(SRC, TGT)
    src_padded = F.pad(SRC, (0, 4), value=PAD)
    tgt_padded = F.pad(TGT, (0, 3), value=PAD)
    out = m(src_padded, tgt_padded)
    torch.testing.assert_close(out[:, :4], base, atol=1e-10, rtol=0)


def test_batch_elements_are_independent() -> None:
    m = model()
    alone = m(SRC[1:], TGT[1:])
    together = m(SRC, TGT)[1:]
    torch.testing.assert_close(together, alone, atol=1e-10, rtol=0)


def test_padded_source_tokens_are_ignored() -> None:
    """A single sentence gives identical outputs whatever its padding width."""
    m = model()
    src_a = torch.tensor([[9, 10, 3, PAD, PAD]])
    src_b = torch.tensor([[9, 10, 3, PAD, PAD, PAD, PAD, PAD]])
    torch.testing.assert_close(m(src_a, TGT[1:]), m(src_b, TGT[1:]), atol=1e-10, rtol=0)


def test_forward_with_attention_returns_detached_maps() -> None:
    m = model(num_encoder_layers=3, num_decoder_layers=2)
    out = m.forward_with_attention(SRC, TGT)
    assert isinstance(out.attention, AttentionMaps)
    torch.testing.assert_close(out.logits, m(SRC, TGT))
    a = out.attention
    assert [t.shape for t in a.encoder_self] == [(2, 4, 5, 5)] * 3
    assert [t.shape for t in a.decoder_self] == [(2, 4, 4, 4)] * 2
    assert [t.shape for t in a.decoder_cross] == [(2, 4, 4, 5)] * 2
    assert all(not t.requires_grad for t in (*a.encoder_self, *a.decoder_cross))
    assert torch.count_nonzero(a.decoder_cross[0][1, ..., 3:]) == 0  # source padding
    assert torch.count_nonzero(torch.triu(a.decoder_self[0], diagonal=1)) == 0  # causal


def test_initialization_follows_adr_011() -> None:
    m = build_transformer(
        small_model_config(d_model=128, num_heads=4, d_ff=256), vocab_size=2000, pad_id=PAD, seed=0
    )
    emb = m.src_embedding.weight
    assert torch.count_nonzero(emb[PAD]) == 0
    assert abs(emb[1:].std().item() - 128**-0.5) < 0.03 * 128**-0.5
    xavier_std = math.sqrt(2.0 / (128 + 128))  # Kaiming default would give ~0.051
    for module in m.modules():
        if isinstance(module, MultiHeadAttention):
            assert abs(module.q_proj.weight.std().item() - xavier_std) < 0.1 * xavier_std
        if isinstance(module, LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))  # type: ignore[arg-type]


def test_reset_parameters_never_applies_linear_defaults() -> None:
    m = build_transformer(small_model_config(d_model=128, d_ff=256), vocab_size=200, pad_id=PAD)
    m.reset_parameters()
    xavier_std = math.sqrt(2.0 / 256)
    q = m.encoder.layers[0].self_attn.q_proj.weight  # type: ignore[union-attr]
    assert abs(q.std().item() - xavier_std) < 0.1 * xavier_std
    assert torch.count_nonzero(m.src_embedding.weight[PAD]) == 0
    assert m.output_weight is m.src_embedding.weight  # still tied


def test_seeded_build_is_reproducible_and_isolated_from_global_rng() -> None:
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    a = build_transformer(small_model_config(), vocab_size=V, pad_id=PAD, seed=7)
    after = torch.rand(1)
    b = build_transformer(small_model_config(), vocab_size=V, pad_id=PAD, seed=7)
    c = build_transformer(small_model_config(), vocab_size=V, pad_id=PAD, seed=8)
    assert torch.equal(before, after)  # global RNG stream untouched by a seeded build
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)
    assert not torch.equal(a.src_embedding.weight, c.src_embedding.weight)


def test_initial_loss_is_near_log_vocab() -> None:
    """math.md §5.3: unit-variance tied logits give CE ≈ ln V + 0.5 at initialization."""
    vocab = 1000
    m = build_transformer(
        small_model_config(d_model=64, d_ff=128), vocab_size=vocab, pad_id=PAD, seed=0
    ).eval()
    g = torch.Generator().manual_seed(0)
    src = torch.randint(4, vocab, (8, 12), generator=g)
    tgt = torch.randint(4, vocab, (8, 10), generator=g)
    with torch.no_grad():
        logits = m(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt[:, 1:].reshape(-1)).item()
    assert math.log(vocab) <= loss <= math.log(vocab) + 1.5


def test_input_validation() -> None:
    m = model()
    with pytest.raises(ValueError, match="batch"):
        m(SRC, TGT[:1])
    with pytest.raises(TypeError):
        m(SRC.float(), TGT)
    with pytest.raises(ValueError):
        m(SRC[0], TGT)
    with pytest.raises(ValueError):
        m(SRC[:, :0], TGT)
    with pytest.raises(ValueError, match="max_len"):
        m(torch.full((1, 65), 5), TGT[:1])  # max_positions = 64
    with pytest.raises(ValueError):
        Transformer(small_model_config(), vocab_size=10, pad_id=10)


def test_debug_checks_catch_invalid_ids() -> None:
    m = model()
    bad = SRC.clone()
    bad[0, 0] = V
    m.debug_checks = True
    with pytest.raises(ValueError, match="IDs span"):
        m(bad, TGT)
    m(SRC, TGT)  # valid input passes all debug checks
