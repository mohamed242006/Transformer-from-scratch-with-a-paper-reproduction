from __future__ import annotations

import pickle

import numpy as np
import pytest

from aiayn.data.copy_task import CopyTaskDataset
from aiayn.data.special_tokens import BOS_ID, EOS_ID, NUM_SPECIAL_TOKENS

KW = {"num_examples": 200, "vocab_size": 12, "min_len": 2, "max_len": 9, "seed": 7}


def test_examples_are_framed_and_in_vocabulary() -> None:
    ds = CopyTaskDataset(**KW)
    for i in range(len(ds)):
        ex = ds[i]
        assert ex.index == i
        assert ex.src[-1] == EOS_ID and ex.tgt[0] == BOS_ID and ex.tgt[-1] == EOS_ID
        content = ex.src[:-1]
        assert content.min() >= NUM_SPECIAL_TOKENS and content.max() < KW["vocab_size"]
        assert KW["min_len"] <= len(content) <= KW["max_len"]


@pytest.mark.parametrize("mode", ["copy", "reverse"])
def test_target_is_copy_or_reverse_of_source(mode: str) -> None:
    ds = CopyTaskDataset(**KW, mode=mode)  # type: ignore[arg-type]
    for i in range(20):
        ex = ds[i]
        src, tgt = ex.src[:-1], ex.tgt[1:-1]
        expected = src if mode == "copy" else src[::-1]
        np.testing.assert_array_equal(tgt, expected)


def test_generation_is_deterministic_and_order_independent() -> None:
    a, b = CopyTaskDataset(**KW), CopyTaskDataset(**KW)
    forward = [a[i].src.tolist() for i in range(len(a))]
    backward = [b[i].src.tolist() for i in reversed(range(len(b)))][::-1]
    assert forward == backward


def test_seed_and_stream_change_the_data() -> None:
    base = CopyTaskDataset(**KW)
    other_seed = CopyTaskDataset(**{**KW, "seed": 8})
    other_stream = CopyTaskDataset(**KW, stream=1)
    first = [base[i].src.tolist() for i in range(20)]
    assert first != [other_seed[i].src.tolist() for i in range(20)]
    assert first != [other_stream[i].src.tolist() for i in range(20)]


def test_length_arrays_match_examples() -> None:
    ds = CopyTaskDataset(**KW)
    for i in range(len(ds)):
        ex = ds[i]
        assert ds.src_lengths[i] == len(ex.src)  # S
        assert ds.tgt_lengths[i] == len(ex.tgt) - 1  # T


def test_index_bounds() -> None:
    ds = CopyTaskDataset(**KW)
    with pytest.raises(IndexError):
        ds[len(ds)]
    with pytest.raises(IndexError):
        ds[-1]


@pytest.mark.parametrize(
    "patch",
    [
        {"num_examples": 0},
        {"vocab_size": 4},
        {"min_len": 0},
        {"min_len": 5, "max_len": 4},
        {"seed": -1},
    ],
)
def test_invalid_arguments(patch: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        CopyTaskDataset(**{**KW, **patch})


def test_pickles_identically() -> None:
    ds = CopyTaskDataset(**KW)
    clone = pickle.loads(pickle.dumps(ds))
    assert all(np.array_equal(ds[i].tgt, clone[i].tgt) for i in range(len(ds)))
