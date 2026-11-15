"""Collation, padding and the teacher-forcing invariants (ADR-009)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from aiayn.data.collate import Batch, Collator, pad_sequences
from aiayn.data.example import Example, frame_source, frame_target
from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID


def make_examples() -> list[Example]:
    pairs = [([5, 6], [7]), ([5, 6, 7, 8], [9, 10, 11]), ([4], [4, 5])]
    return [
        Example(index=10 + i, src=frame_source(s), tgt=frame_target(t))
        for i, (s, t) in enumerate(pairs)
    ]


@pytest.fixture
def batch() -> Batch:
    return Collator()(make_examples())


def test_shapes_and_dtypes(batch: Batch) -> None:
    assert batch.src.shape == (3, 5)  # longest source: 4 tokens + EOS
    assert batch.tgt_in.shape == batch.tgt_out.shape == (3, 4)  # longest target: 3 + 1
    for t in (batch.src, batch.tgt_in, batch.tgt_out, batch.src_lengths, batch.indices):
        assert t.dtype == torch.long
    assert batch.batch_size == 3
    assert batch.indices.tolist() == [10, 11, 12]


def test_exact_contents(batch: Batch) -> None:
    assert batch.src.tolist() == [
        [5, 6, EOS_ID, PAD_ID, PAD_ID],
        [5, 6, 7, 8, EOS_ID],
        [4, EOS_ID, PAD_ID, PAD_ID, PAD_ID],
    ]
    assert batch.tgt_in.tolist() == [
        [BOS_ID, 7, PAD_ID, PAD_ID],
        [BOS_ID, 9, 10, 11],
        [BOS_ID, 4, 5, PAD_ID],
    ]
    assert batch.tgt_out.tolist() == [
        [7, EOS_ID, PAD_ID, PAD_ID],
        [9, 10, 11, EOS_ID],
        [4, 5, EOS_ID, PAD_ID],
    ]


def test_teacher_forcing_invariants(batch: Batch) -> None:
    tgt_in, tgt_out = batch.tgt_in, batch.tgt_out
    # Identical padding patterns on both sides of the shift.
    assert torch.equal(tgt_in == PAD_ID, tgt_out == PAD_ID)
    # out[t] == in[t+1] wherever both are real tokens.
    both = (tgt_in[:, 1:] != PAD_ID) & (tgt_out[:, :-1] != PAD_ID)
    assert torch.equal(tgt_out[:, :-1][both], tgt_in[:, 1:][both])
    # Decoder input never contains EOS; it always starts with BOS.
    assert not (tgt_in == EOS_ID).any()
    assert (tgt_in[:, 0] == BOS_ID).all()
    # Each row's prediction target ends with EOS at position length - 1.
    rows = torch.arange(batch.batch_size)
    assert (tgt_out[rows, batch.tgt_lengths - 1] == EOS_ID).all()


def test_token_counts(batch: Batch) -> None:
    assert batch.num_tgt_tokens == int((batch.tgt_out != PAD_ID).sum()) == 9
    assert batch.num_src_tokens == int((batch.src != PAD_ID).sum()) == 10
    assert batch.src_lengths.tolist() == [3, 5, 2]
    assert batch.tgt_lengths.tolist() == [2, 4, 3]


def test_pad_to_multiple_of() -> None:
    b = Collator(pad_to_multiple_of=8)(make_examples())
    assert b.src.shape == (3, 8) and b.tgt_in.shape == (3, 8)
    assert b.num_tgt_tokens == 9  # extra padding is not counted


def test_pad_sequences_validation() -> None:
    with pytest.raises(ValueError):
        pad_sequences([])
    with pytest.raises(ValueError):
        pad_sequences([np.array([1], dtype=np.int32)], pad_to_multiple_of=0)


def test_empty_batch_rejected() -> None:
    with pytest.raises(ValueError):
        Collator()([])


def test_batch_to_device_and_pickle(batch: Batch) -> None:
    moved = batch.to("cpu")
    assert torch.equal(moved.src, batch.src) and moved.num_tgt_tokens == batch.num_tgt_tokens
    assert isinstance(pickle.loads(pickle.dumps(Collator())), Collator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning requires an accelerator")
def test_pin_memory(batch: Batch) -> None:
    assert batch.pin_memory().src.is_pinned()
