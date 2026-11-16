"""Greedy and beam search (ADR-019): reference implementations and exhaustive search."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID
from aiayn.inference.decoding import (
    DecodeConfig,
    beam_search,
    greedy_decode,
    length_penalty,
)
from aiayn.models import Transformer, build_transformer
from tests.conftest import small_model_config

SRC = torch.tensor([[5, 6, 7, 3, PAD_ID], [8, 9, 10, 11, 3], [4, 3, PAD_ID, PAD_ID, PAD_ID]])


def model(vocab: int = 12, seed: int = 0) -> Transformer:
    m = build_transformer(
        small_model_config(max_positions=64), vocab_size=vocab, pad_id=PAD_ID, seed=seed
    ).double()
    return m.eval()


@torch.no_grad()
def next_log_probs(m: Transformer, src: torch.Tensor, prefix: list[int]) -> torch.Tensor:
    memory, mask, _ = m.encode(src)
    hidden, _, _ = m.decode(torch.tensor([[BOS_ID, *prefix]]), memory, mask)
    lp = F.log_softmax(m.generator(hidden[:, -1]).double(), dim=-1)[0]
    lp[PAD_ID] = lp[BOS_ID] = -math.inf
    return lp


def reference_greedy(m: Transformer, src: torch.Tensor, limit: int) -> list[int]:
    """One sentence, no batching, no tricks."""
    out: list[int] = []
    for step in range(1, limit + 1):
        lp = next_log_probs(m, src, out)
        if step == limit:
            return out  # EOS forced
        token = int(lp.argmax())
        if token == EOS_ID:
            return out
        out.append(token)
    return out


def test_greedy_matches_unbatched_reference() -> None:
    m = model()
    config = DecodeConfig(beam_size=1, max_len_a=0.0, max_len_b=6)
    hyps = greedy_decode(m, SRC, config)
    for i in range(SRC.shape[0]):
        src = SRC[i : i + 1, : int((SRC[i] != PAD_ID).sum())]
        assert hyps[i].tokens == reference_greedy(m, src, 6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_beam_of_one_equals_greedy_without_length_penalty(seed: int) -> None:
    m = model(seed=seed)
    config = DecodeConfig(beam_size=1, length_penalty=0.0, max_len_a=1.0, max_len_b=5)
    greedy = greedy_decode(m, SRC, config)
    beam = beam_search(m, SRC, config)
    assert [h.tokens for h in beam] == [h.tokens for h in greedy]
    for g, b in zip(greedy, beam, strict=True):
        assert b.score == pytest.approx(g.score, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.0, 0.6, 1.0])
def test_wide_beam_equals_exhaustive_search(alpha: float) -> None:
    """With k at least the number of hypotheses, beam search is exhaustive: it must return
    the sequence with the best length-normalized score among all of them."""
    m = model(vocab=7, seed=3)  # specials 0-3, content {UNK=1, 4, 5, 6}
    src = torch.tensor([[4, 5, 3]])
    limit = 3  # at most 2 content tokens, then EOS
    config = DecodeConfig(beam_size=64, length_penalty=alpha, max_len_a=0.0, max_len_b=limit)
    content = [1, 4, 5, 6]
    best_seq, best_norm = None, -math.inf
    for n in range(limit):
        for seq in itertools.product(content, repeat=n):
            score = sum(float(next_log_probs(m, src, list(seq[:t]))[seq[t]]) for t in range(n))
            score += float(next_log_probs(m, src, list(seq))[EOS_ID])
            norm = score / length_penalty(n + 1, alpha)
            if norm > best_norm:
                best_seq, best_norm = list(seq), norm
    hyp = beam_search(m, src, config)[0]
    assert hyp.tokens == best_seq
    assert hyp.normalized_score == pytest.approx(best_norm, abs=1e-9)


def test_batched_beam_equals_one_sentence_at_a_time() -> None:
    m = model()
    config = DecodeConfig(beam_size=4, max_len_b=8)
    batched = beam_search(m, SRC, config)
    for i in range(SRC.shape[0]):
        src = SRC[i : i + 1, : int((SRC[i] != PAD_ID).sum())]
        alone = beam_search(m, src, config)[0]
        assert alone.tokens == batched[i].tokens
        assert alone.score == pytest.approx(batched[i].score, abs=1e-9)


def test_length_limits_and_special_tokens() -> None:
    m = model()
    for decode, config in (
        (greedy_decode, DecodeConfig(beam_size=1, max_len_a=0.0, max_len_b=3)),
        (beam_search, DecodeConfig(beam_size=3, max_len_a=0.0, max_len_b=3)),
    ):
        for hyp in decode(m, SRC, config):
            assert len(hyp.tokens) <= 2  # 3 generated tokens include the forced EOS
            assert not {PAD_ID, BOS_ID, EOS_ID} & set(hyp.tokens)
            assert math.isfinite(hyp.score) and hyp.score <= 0


def test_positional_table_caps_the_output_length() -> None:
    m = build_transformer(
        small_model_config(max_positions=6), vocab_size=12, pad_id=PAD_ID, seed=0
    ).eval()
    hyps = greedy_decode(m, SRC[:, :4], DecodeConfig(beam_size=1, max_len_b=50))
    assert all(len(h.tokens) <= 4 for h in hyps)  # max_positions - 1 tokens incl. EOS


def test_length_penalty_formula() -> None:
    assert length_penalty(1, 0.6) == pytest.approx(1.0)
    assert length_penalty(7, 1.0) == pytest.approx(2.0)
    assert length_penalty(10, 0.0) == 1.0


def test_decode_config_validation() -> None:
    with pytest.raises(ValueError):
        DecodeConfig(beam_size=0)
    with pytest.raises(ValueError):
        DecodeConfig(max_len_b=0)
