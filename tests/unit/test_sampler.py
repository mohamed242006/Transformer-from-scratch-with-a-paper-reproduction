"""Token-budget batch sampler (ADR-009)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aiayn.data.sampler import TokenBudgetBatchSampler


def make_lengths(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.integers(1, 60, size=n)
    tgt = np.clip(src + rng.integers(-5, 6, size=n), 1, None)
    return src, tgt


def check_epoch(sampler: TokenBudgetBatchSampler, src: np.ndarray, tgt: np.ndarray) -> None:
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(src))), "each index exactly once"
    sizes = np.maximum(src, tgt)
    for b in batches:
        assert b, "no empty batches"
        assert len(b) * int(sizes[b].max()) <= sampler.max_tokens
        if sampler.max_sentences is not None:
            assert len(b) <= sampler.max_sentences
    assert len(sampler) == len(batches)


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("max_sentences", [None, 7])
def test_epoch_invariants(shuffle: bool, max_sentences: int | None) -> None:
    src, tgt = make_lengths()
    sampler = TokenBudgetBatchSampler(
        src, tgt, max_tokens=256, max_sentences=max_sentences, shuffle=shuffle, pool_size=100
    )
    check_epoch(sampler, src, tgt)


def test_deterministic_per_seed_and_epoch_and_varies_across_epochs() -> None:
    src, tgt = make_lengths()
    a = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=3, pool_size=100)
    b = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=3, pool_size=100)
    assert list(a) == list(b)
    epoch0 = list(a)
    a.set_epoch(1)
    assert list(a) != epoch0
    a.set_epoch(0)
    assert list(a) == epoch0
    c = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=4, pool_size=100)
    assert list(c) != epoch0


def test_eval_order_is_sorted_and_deterministic() -> None:
    src, tgt = make_lengths()
    sampler = TokenBudgetBatchSampler(src, tgt, max_tokens=256, shuffle=False)
    flat = [i for b in sampler for i in b]
    sizes = np.maximum(src, tgt)[flat]
    assert (np.diff(sizes) >= 0).all()
    sampler.set_epoch(5)
    assert [i for b in sampler for i in b] == flat


def test_bucketing_reduces_padding() -> None:
    src, tgt = make_lengths(2000)
    sampler = TokenBudgetBatchSampler(src, tgt, max_tokens=512, pool_size=1000)
    # Random batches of the same count would give roughly mean/max ≈ 0.5 here.
    assert sampler.padding_efficiency() > 0.9


def test_oversized_examples_are_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="longer than max_tokens"):
        TokenBudgetBatchSampler([10, 300], [10, 10], max_tokens=256)


@pytest.mark.parametrize(
    ("src", "tgt", "kwargs"),
    [
        ([], [], {}),
        ([1, 2], [1], {}),
        ([0, 2], [1, 2], {}),
        ([1], [1], {"max_tokens": 0}),
        ([1], [1], {"pool_size": 0}),
        ([1], [1], {"max_sentences": 0}),
        ([1], [1], {"seed": -1}),
    ],
)
def test_invalid_arguments(src: list[int], tgt: list[int], kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        TokenBudgetBatchSampler(src, tgt, **{"max_tokens": 64, **kwargs})


def test_negative_epoch_rejected() -> None:
    sampler = TokenBudgetBatchSampler([1], [1], max_tokens=8)
    with pytest.raises(ValueError):
        sampler.set_epoch(-1)


@settings(max_examples=60, deadline=None)
@given(
    lengths=st.lists(st.tuples(st.integers(1, 40), st.integers(1, 40)), min_size=1, max_size=200),
    max_tokens=st.integers(40, 400),
    pool_size=st.integers(1, 64),
    max_sentences=st.one_of(st.none(), st.integers(1, 16)),
    shuffle=st.booleans(),
    seed=st.integers(0, 2**16),
)
def test_property_every_index_once_within_budget(
    lengths: list[tuple[int, int]],
    max_tokens: int,
    pool_size: int,
    max_sentences: int | None,
    shuffle: bool,
    seed: int,
) -> None:
    src = np.array([s for s, _ in lengths])
    tgt = np.array([t for _, t in lengths])
    sampler = TokenBudgetBatchSampler(
        src,
        tgt,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        shuffle=shuffle,
        pool_size=pool_size,
        seed=seed,
    )
    check_epoch(sampler, src, tgt)


# ---------------------------------------------------------------------- #
# Mid-epoch resumption (ADR-013)
# ---------------------------------------------------------------------- #
def test_resuming_mid_epoch_yields_the_remaining_batches() -> None:
    src, tgt = make_lengths()
    full = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=5, pool_size=100)
    full.set_epoch(2)
    expected = list(full)
    resumed = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=5, pool_size=100)
    resumed.set_epoch(2, start_batch=7)
    assert list(resumed) == expected[7:]
    assert len(resumed) == len(expected) - 7
    resumed.set_epoch(3)  # the next epoch starts from the beginning again
    assert resumed.start_batch == 0


def test_state_dict_round_trip_restores_position() -> None:
    src, tgt = make_lengths()
    a = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=5)
    a.set_epoch(4, start_batch=3)
    b = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=5)
    b.load_state_dict(a.state_dict())
    assert (b.epoch, b.start_batch) == (4, 3) and list(b) == list(a)


def test_state_dict_refuses_different_batching_settings() -> None:
    src, tgt = make_lengths()
    state = TokenBudgetBatchSampler(src, tgt, max_tokens=256, seed=5).state_dict()
    other = TokenBudgetBatchSampler(src, tgt, max_tokens=512, seed=5)
    with pytest.raises(ValueError, match="max_tokens"):
        other.load_state_dict(state)


def test_invalid_start_batch() -> None:
    src, tgt = make_lengths(50)
    sampler = TokenBudgetBatchSampler(src, tgt, max_tokens=256)
    with pytest.raises(ValueError):
        sampler.set_epoch(0, start_batch=-1)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.set_epoch(0, start_batch=len(sampler) + 1)
    sampler.set_epoch(0, start_batch=len(sampler))  # exactly exhausted is valid
    assert list(sampler) == []
