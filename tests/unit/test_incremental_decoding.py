"""Incremental decoding (ADR-023 §6): every model's cached step equals full-prefix decoding,
and greedy/beam search give identical hypotheses with and without the cache."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from aiayn.inference.decoding import DecodeConfig, beam_search, greedy_decode
from aiayn.models import EncoderDecoder, RNNConfig, build_model, build_transformer
from tests.conftest import small_model_config

V = 40


def rnn(cell: str, attention: str, bi: bool) -> EncoderDecoder:
    config = RNNConfig(
        name="t",
        architecture="rnn",
        cell=cell,  # type: ignore[arg-type]
        attention=attention,  # type: ignore[arg-type]
        embedding_dim=16,
        hidden_size=24,
        num_layers=2,
        bidirectional_encoder=bi,
        dropout=0.0,
    )
    return build_model(config, vocab_size=V, pad_id=0, seed=3)


MODELS: dict[str, Any] = {
    "post_ln": lambda: build_transformer(
        small_model_config(max_positions=128), vocab_size=V, pad_id=0, seed=1
    ),
    "pre_ln": lambda: build_transformer(
        small_model_config(max_positions=128, norm_position="pre"), vocab_size=V, pad_id=0, seed=2
    ),
    "lstm": lambda: rnn("lstm", "none", False),
    "gru_bahdanau": lambda: rnn("gru", "bahdanau", True),
    "lstm_luong": lambda: rnn("lstm", "luong", True),
}


def batch() -> torch.Tensor:
    torch.manual_seed(0)
    src = torch.randint(4, V, (5, 9))
    src[:, -1] = 3
    src[3, 5:] = 0
    src[3, 4] = 3
    return src


@pytest.mark.parametrize("name", list(MODELS))
def test_steps_equal_full_prefix_decoding(name: str) -> None:
    model = MODELS[name]().double().eval()
    assert model.supports_incremental
    src = batch()
    tgt = torch.randint(4, V, (5, 7))
    tgt[:, 0] = 2
    with torch.no_grad():
        memory, mask, _ = model.encode(src)
        full, _, _ = model.decode(tgt, memory, mask)
        state = model.init_state(memory, mask)
        for t in range(tgt.shape[1]):
            hidden, state = model.decode_step(tgt[:, t], state)
            torch.testing.assert_close(hidden, full[:, t], rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", list(MODELS))
def test_reorder_selects_rows(name: str) -> None:
    model = MODELS[name]().double().eval()
    src = batch()
    tokens = torch.tensor([2, 5, 6, 7, 8])
    rows = torch.tensor([4, 4, 0, 2, 1])
    with torch.no_grad():
        memory, mask, _ = model.encode(src)
        state = model.init_state(memory, mask)
        _, state = model.decode_step(tokens, state)
        state = model.reorder_state(state, rows)
        after, _ = model.decode_step(torch.full((5,), 9), state)
        memory2, mask2, _ = model.encode(src[rows])
        reference = model.init_state(memory2, mask2)
        _, reference = model.decode_step(tokens[rows], reference)
        expected, _ = model.decode_step(torch.full((5,), 9), reference)
    torch.testing.assert_close(after, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize(
    "config",
    [
        DecodeConfig(beam_size=1, max_len_b=15),
        DecodeConfig(beam_size=4, max_len_b=15),
        DecodeConfig(beam_size=5, length_penalty=1.0, max_len_b=15),
    ],
    ids=["greedy", "beam4", "beam5_a1"],
)
def test_search_is_identical_with_and_without_cache(name: str, config: DecodeConfig) -> None:
    model = MODELS[name]().double().eval()
    search = greedy_decode if config.beam_size == 1 else beam_search
    cached = search(model, batch(), config, incremental=True)
    prefix = search(model, batch(), config, incremental=False)
    assert [h.tokens for h in cached] == [h.tokens for h in prefix]
    for a, b in zip(cached, prefix, strict=True):
        assert a.score == pytest.approx(b.score, abs=1e-9)
