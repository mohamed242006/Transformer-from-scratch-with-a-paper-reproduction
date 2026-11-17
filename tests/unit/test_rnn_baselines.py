"""RNN baselines (ADR-021): configs, parameter counts, the EncoderDecoder contract
(causality, padding invariance, beam expansion), attention, decoding and overfitting."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import torch

from aiayn.data.special_tokens import PAD_ID
from aiayn.inference.decoding import DecodeConfig, beam_search, greedy_decode
from aiayn.models import (
    ModelConfig,
    RNNConfig,
    RNNSeq2Seq,
    analytic_parameter_count,
    build_model,
    load_any_model_config,
    model_config_from_dict,
)
from aiayn.models.baselines import BahdanauAttention, LuongAttention, attend
from aiayn.training.losses import label_smoothed_nll
from aiayn.utils.config import ConfigError, config_to_dict
from aiayn.utils.params import count_parameters

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "model"
V = 30
VARIANTS = [
    ("lstm", "none", False),
    ("gru", "bahdanau", True),
    ("lstm", "luong", True),
    ("gru", "luong", False),
    ("lstm", "bahdanau", False),
    ("gru", "none", True),
]


def rnn_config(
    cell: str = "lstm", attention: str = "luong", bi: bool = True, **kw: Any
) -> RNNConfig:
    base: dict[str, Any] = {
        "name": "t",
        "architecture": "rnn",
        "cell": cell,
        "attention": attention,
        "embedding_dim": 16,
        "hidden_size": 24,
        "num_layers": 2,
        "bidirectional_encoder": bi,
        "dropout": 0.0,
        "max_positions": 64,
    }
    base.update(kw)
    return RNNConfig(**base)


def model(cell: str = "lstm", attention: str = "luong", bi: bool = True, **kw: Any) -> RNNSeq2Seq:
    m = build_model(rnn_config(cell, attention, bi, **kw), vocab_size=V, pad_id=PAD_ID, seed=0)
    assert isinstance(m, RNNSeq2Seq)
    return m


SRC = torch.tensor([[5, 6, 7, 8, 3], [9, 10, 3, 0, 0], [11, 3, 0, 0, 0]])
TGT = torch.tensor([[2, 12, 13, 14], [2, 15, 0, 0], [2, 16, 17, 0]])


# ---------------------------------------------------------------------------- configs
@pytest.mark.parametrize("name", ["lstm_seq2seq", "gru_bahdanau", "bilstm_luong"])
def test_baseline_configs_match_the_transformer_within_ten_percent(name: str) -> None:
    cfg = load_any_model_config(CONFIGS / f"{name}.yaml")
    assert isinstance(cfg, RNNConfig) and cfg.name == name
    reference = analytic_parameter_count(
        load_any_model_config(CONFIGS / "transformer_tiny.yaml"), 8000
    )
    ours = analytic_parameter_count(cfg, 8000)
    assert abs(ours / reference - 1) < 0.02  # protocol: ±10%; configs are sized within 2%
    m = build_model(cfg, vocab_size=8000, pad_id=PAD_ID, seed=0)
    assert count_parameters(m) == ours


@pytest.mark.parametrize(("cell", "attention", "bi"), VARIANTS)
@pytest.mark.parametrize("tied", [True, False])
def test_analytic_parameter_count(cell: str, attention: str, bi: bool, tied: bool) -> None:
    for layers in (1, 3):
        cfg = rnn_config(cell, attention, bi, num_layers=layers, tie_weights=tied)
        m = build_model(cfg, vocab_size=V, pad_id=PAD_ID, seed=0)
        assert count_parameters(m) == analytic_parameter_count(cfg, V)


def test_config_dispatch_keeps_transformer_configs_unchanged() -> None:
    transformer = load_any_model_config(CONFIGS / "transformer_tiny.yaml")
    assert isinstance(transformer, ModelConfig)
    assert "architecture" not in config_to_dict(transformer)  # config hash unchanged
    rnn = rnn_config()
    assert model_config_from_dict(config_to_dict(rnn)) == rnn
    overridden = load_any_model_config(CONFIGS / "bilstm_luong.yaml", ("hidden_size=64",))
    assert isinstance(overridden, RNNConfig) and overridden.hidden_size == 64


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"name": "Bad"}, "snake_case"),
        ({"dropout": 1.0}, "dropout"),
        ({"hidden_size": 0}, "positive"),
    ],
)
def test_config_validation(override: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        rnn_config(**override)


def test_unknown_cell_is_rejected() -> None:
    raw = config_to_dict(rnn_config())
    raw["cell"] = "rnn"
    with pytest.raises(ConfigError):
        model_config_from_dict(raw)


# ---------------------------------------------------------------------------- contract
@pytest.mark.parametrize(("cell", "attention", "bi"), VARIANTS)
def test_shapes_causality_and_padding_invariance(cell: str, attention: str, bi: bool) -> None:
    m = model(cell, attention, bi).double().eval()
    logits = m(SRC, TGT)
    assert logits.shape == (3, 4, V) and logits.dtype == torch.float64
    changed = TGT.clone()
    changed[:, 3] = 20  # the future must not affect earlier positions
    assert torch.equal(m(SRC, changed)[:, :3], logits[:, :3])
    for row in range(3):  # padding (and batch companions) must not change a sentence
        s_len, t_len = int((SRC[row] != 0).sum()), int((TGT[row] != 0).sum())
        alone = m(SRC[row : row + 1, :s_len], TGT[row : row + 1, :t_len])
        torch.testing.assert_close(alone, logits[row : row + 1, :t_len], rtol=0, atol=1e-12)


@pytest.mark.parametrize(("cell", "attention", "bi"), VARIANTS)
def test_memory_repeat_interleave_matches_repeated_sources(
    cell: str, attention: str, bi: bool
) -> None:
    m = model(cell, attention, bi).double().eval()
    memory, mask, _ = m.encode(SRC)
    expanded = memory.repeat_interleave(2, dim=0)
    direct, direct_mask, _ = m.encode(SRC.repeat_interleave(2, dim=0))
    tgt = TGT.repeat_interleave(2, dim=0)
    a, _, _ = m.decode(tgt, expanded, mask.repeat_interleave(2, dim=0))
    b, _, _ = m.decode(tgt, direct, direct_mask)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="batch"):
        memory.repeat_interleave(2, dim=1)


def test_attention_maps_are_distributions_over_real_tokens() -> None:
    for attention in ("bahdanau", "luong"):
        m = model(attention=attention).double().eval()
        memory, mask, _ = m.encode(SRC)
        _, _, (weights,) = m.decode(TGT, memory, mask, need_weights=True)
        assert weights.shape == (3, 4, 5)
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 4, dtype=torch.float64))
        assert torch.all(weights.masked_select(~mask[:, None, :].expand_as(weights)) == 0)
    plain = model(attention="none").eval()
    memory, mask, _ = plain.encode(SRC)
    assert plain.decode(TGT, memory, mask, need_weights=True)[2] == ()


def test_attend_masks_and_promotes_half_precision() -> None:
    scores = torch.randn(2, 3, 4, dtype=torch.float16)
    values = torch.randn(2, 4, 5, dtype=torch.float16)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    context, weights = attend(scores, values, mask)
    assert weights.dtype == torch.float32 and context.dtype == torch.float16
    assert torch.all(weights[0, :, 2:] == 0)


@pytest.mark.parametrize("cls", [BahdanauAttention, LuongAttention])
def test_attention_gradcheck(cls: type) -> None:
    torch.manual_seed(0)
    module = (cls(6, 8, 5) if cls is BahdanauAttention else cls(6, 8)).double()
    query = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    values = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    def fn(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return module(q, module.project_keys(v), v, mask)[0]

    assert torch.autograd.gradcheck(fn, (query, values))


def test_seeded_build_is_deterministic_and_leaves_global_rng() -> None:
    torch.manual_seed(123)
    before = torch.get_rng_state()
    a, b = model(), model()
    assert torch.equal(torch.get_rng_state(), before)
    assert all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )
    assert torch.all(a.src_embedding.weight[PAD_ID] == 0)
    assert all(p.abs().max() <= 0.1 for p in a.parameters())


def test_tying() -> None:
    tied, untied = model(), model(tie_weights=False)
    assert tied.output_projection is None and tied.output_weight is tied.src_embedding.weight
    assert untied.src_embedding is not untied.tgt_embedding
    assert untied.output_weight is untied.output_projection


@pytest.mark.parametrize(
    ("bad_src", "error"),
    [(torch.zeros(3, dtype=torch.long), ValueError), (torch.zeros(1, 3), TypeError)],
)
def test_input_validation(bad_src: torch.Tensor, error: type) -> None:
    with pytest.raises(error):
        model().encode(bad_src)


# ---------------------------------------------------------------------------- decoding
@pytest.mark.parametrize(("cell", "attention", "bi"), VARIANTS[:3])
def test_shared_decoders_work_and_beam_one_equals_greedy(
    cell: str, attention: str, bi: bool
) -> None:
    m = model(cell, attention, bi)
    greedy = greedy_decode(m, SRC, DecodeConfig(beam_size=1, max_len_b=8))
    beam1 = beam_search(m, SRC, DecodeConfig(beam_size=1, length_penalty=0.0, max_len_b=8))
    assert [h.tokens for h in greedy] == [h.tokens for h in beam1]
    beam4 = beam_search(m, SRC, DecodeConfig(beam_size=4, max_len_b=8))
    assert len(beam4) == 3
    assert all(h.normalized_score >= -1e9 and PAD_ID not in h.tokens for h in beam4)


# ---------------------------------------------------------------------------- learning
@pytest.mark.parametrize(
    ("cell", "attention", "bi"),
    [("lstm", "none", False), ("gru", "bahdanau", True), ("lstm", "luong", True)],
)
def test_overfits_one_batch(cell: str, attention: str, bi: bool) -> None:
    torch.manual_seed(0)
    src = torch.randint(4, V, (6, 7))
    src[:, -1] = 3
    tgt_in = torch.cat([torch.full((6, 1), 2), src[:, :-1]], dim=1)
    tgt_out = src.clone()
    m = model(cell, attention, bi, embedding_dim=32, hidden_size=48, num_layers=1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        out = label_smoothed_nll(m(src, tgt_in), tgt_out, epsilon=0.0, ignore_index=PAD_ID)
        loss = out.loss_sum / tgt_out.numel()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.02, (cell, attention, loss.item())


def test_variants_are_distinct_architectures() -> None:
    names = {type(model(c, a, b).attention).__name__ for c, a, b in itertools.islice(VARIANTS, 3)}
    assert names == {"NoneType", "BahdanauAttention", "LuongAttention"}
