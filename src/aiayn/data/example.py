"""Sequence framing and teacher forcing (ADR-009).

Framing convention::

    source  : x_1 ... x_n EOS          (length S = n + 1)
    target  : BOS y_1 ... y_m EOS      (length m + 2)

    teacher forcing (per example, before padding):
    tgt_in  : BOS y_1 ... y_m          (length T = m + 1)
    tgt_out : y_1 ... y_m EOS          (length T = m + 1)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aiayn.data.special_tokens import BOS_ID, EOS_ID, PAD_ID

__all__ = [
    "TOKEN_DTYPE",
    "Example",
    "TokenArray",
    "frame_source",
    "frame_target",
    "teacher_forcing_split",
]

#: Storage dtype for token IDs before collation (vocabularies are far below 2**31).
TOKEN_DTYPE = np.int32
TokenArray = npt.NDArray[np.int32]

_FORBIDDEN_IN_CONTENT = np.array([PAD_ID, BOS_ID, EOS_ID], dtype=np.int64)


@dataclass(frozen=True)
class Example:
    """One framed sequence pair.

    Attributes:
        index: Position of the example in its dataset (used to restore corpus order).
        src: Framed source ``x_1..x_n EOS``, shape ``[S]``.
        tgt: Framed target ``BOS y_1..y_m EOS``, shape ``[m + 2]``.
    """

    index: int
    src: TokenArray
    tgt: TokenArray


def _as_content(ids: Sequence[int] | npt.NDArray[np.integer]) -> TokenArray:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"token sequence must be 1-D, got shape {arr.shape}")
    if arr.size:
        if int(np.min(arr)) < 0 or int(np.max(arr)) > np.iinfo(TOKEN_DTYPE).max:
            raise ValueError("token IDs out of range")
        if np.isin(arr, _FORBIDDEN_IN_CONTENT).any():
            raise ValueError("content must not contain PAD/BOS/EOS IDs (UNK is allowed)")
    return arr.astype(TOKEN_DTYPE)


def frame_source(ids: Sequence[int] | npt.NDArray[np.integer]) -> TokenArray:
    """``x_1..x_n`` -> ``x_1..x_n EOS``."""
    content = _as_content(ids)
    return np.concatenate([content, np.array([EOS_ID], dtype=TOKEN_DTYPE)])


def frame_target(ids: Sequence[int] | npt.NDArray[np.integer]) -> TokenArray:
    """``y_1..y_m`` -> ``BOS y_1..y_m EOS``."""
    content = _as_content(ids)
    bos = np.array([BOS_ID], dtype=TOKEN_DTYPE)
    eos = np.array([EOS_ID], dtype=TOKEN_DTYPE)
    return np.concatenate([bos, content, eos])


def teacher_forcing_split(target: TokenArray) -> tuple[TokenArray, TokenArray]:
    """Split a framed target into decoder input and prediction target.

    ``BOS y_1..y_m EOS`` -> (``BOS y_1..y_m``, ``y_1..y_m EOS``). Both have length m + 1,
    and ``out[t] == in[t + 1]`` for every ``t < m``.
    """
    if target.ndim != 1 or target.shape[0] < 2:
        raise ValueError(f"framed target must be 1-D with length >= 2, got {target.shape}")
    if int(target[0]) != BOS_ID or int(target[-1]) != EOS_ID:
        raise ValueError("framed target must start with BOS and end with EOS")
    return target[:-1], target[1:]
