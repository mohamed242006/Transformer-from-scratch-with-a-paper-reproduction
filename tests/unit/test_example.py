"""Framing and teacher forcing (ADR-009)."""

from __future__ import annotations

import numpy as np
import pytest

from aiayn.data.example import (
    TOKEN_DTYPE,
    frame_source,
    frame_target,
    teacher_forcing_split,
)
from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID, UNK_ID


def test_source_framing_appends_eos_only() -> None:
    out = frame_source([7, 8, 9])
    assert out.tolist() == [7, 8, 9, EOS_ID]
    assert out.dtype == TOKEN_DTYPE


def test_target_framing_wraps_with_bos_eos() -> None:
    assert frame_target([7, 8]).tolist() == [BOS_ID, 7, 8, EOS_ID]


def test_empty_content_is_framed() -> None:
    assert frame_source([]).tolist() == [EOS_ID]
    assert frame_target([]).tolist() == [BOS_ID, EOS_ID]


@pytest.mark.parametrize("special", [PAD_ID, BOS_ID, EOS_ID])
def test_content_may_not_contain_structural_specials(special: int) -> None:
    with pytest.raises(ValueError, match="PAD/BOS/EOS"):
        frame_source([5, special, 6])


def test_unk_is_allowed_in_content() -> None:
    assert frame_target([UNK_ID]).tolist() == [BOS_ID, UNK_ID, EOS_ID]


@pytest.mark.parametrize("bad", [[-1], [[4, 5]], [2**31]])
def test_invalid_token_arrays_are_rejected(bad: list[object]) -> None:
    with pytest.raises(ValueError):
        frame_source(bad)  # type: ignore[arg-type]


def test_teacher_forcing_is_a_one_step_shift() -> None:
    tgt = frame_target([7, 8, 9])
    tgt_in, tgt_out = teacher_forcing_split(tgt)
    assert tgt_in.tolist() == [BOS_ID, 7, 8, 9]
    assert tgt_out.tolist() == [7, 8, 9, EOS_ID]
    assert len(tgt_in) == len(tgt_out)
    np.testing.assert_array_equal(tgt_out[:-1], tgt_in[1:])  # out[t] == in[t+1]
    assert EOS_ID not in tgt_in and BOS_ID not in tgt_out


@pytest.mark.parametrize("bad", [[BOS_ID], [7, 8, EOS_ID], [BOS_ID, 7, 8]])
def test_teacher_forcing_rejects_unframed_targets(bad: list[int]) -> None:
    with pytest.raises(ValueError):
        teacher_forcing_split(np.asarray(bad, dtype=TOKEN_DTYPE))
