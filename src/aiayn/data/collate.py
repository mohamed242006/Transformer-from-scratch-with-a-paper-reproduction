"""Batch collation: padding and teacher-forcing split (ADR-009).

Shapes (B = batch size, S = padded source length, T = padded decoder length)::

    src      [B, S] long   x_1..x_n EOS PAD...
    tgt_in   [B, T] long   BOS y_1..y_m PAD...
    tgt_out  [B, T] long   y_1..y_m EOS PAD...
    src_lengths, tgt_lengths, indices  [B] long

Masks are NOT built here; see ``aiayn.modules.masking`` (ADR-008).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from aiayn.data.example import Example, TokenArray, teacher_forcing_split
from aiayn.data.special_tokens import PAD_ID

__all__ = ["Batch", "Collator", "pad_sequences"]


@dataclass(frozen=True)
class Batch:
    """A padded mini-batch. All tensors live on the same device."""

    src: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    src_lengths: torch.Tensor
    tgt_lengths: torch.Tensor
    indices: torch.Tensor
    #: Non-pad tokens in ``tgt_out`` (= sum of tgt_lengths), the loss normalizer.
    #: Stored as a Python int so reading it never forces a GPU sync.
    num_tgt_tokens: int
    num_src_tokens: int

    @property
    def batch_size(self) -> int:
        return int(self.src.shape[0])

    def to(self, device: torch.device | str, non_blocking: bool = False) -> Batch:
        """Move every tensor to ``device``."""
        return replace(
            self,
            src=self.src.to(device, non_blocking=non_blocking),
            tgt_in=self.tgt_in.to(device, non_blocking=non_blocking),
            tgt_out=self.tgt_out.to(device, non_blocking=non_blocking),
            src_lengths=self.src_lengths.to(device, non_blocking=non_blocking),
            tgt_lengths=self.tgt_lengths.to(device, non_blocking=non_blocking),
            indices=self.indices.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> Batch:
        """Called by ``DataLoader(pin_memory=True)`` for custom batch types."""
        return replace(
            self,
            src=self.src.pin_memory(),
            tgt_in=self.tgt_in.pin_memory(),
            tgt_out=self.tgt_out.pin_memory(),
            src_lengths=self.src_lengths.pin_memory(),
            tgt_lengths=self.tgt_lengths.pin_memory(),
            indices=self.indices.pin_memory(),
        )


def pad_sequences(
    seqs: Sequence[TokenArray], pad_id: int = PAD_ID, pad_to_multiple_of: int = 1
) -> torch.Tensor:
    """Right-pad 1-D token arrays into a ``[B, L]`` long tensor.

    ``L`` is the longest length, rounded up to a multiple of ``pad_to_multiple_of``
    (e.g. 8 for tensor-core-friendly shapes under mixed precision).
    """
    if not seqs:
        raise ValueError("cannot pad an empty list of sequences")
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be >= 1")
    longest = max(int(s.shape[0]) for s in seqs)
    width = -(-longest // pad_to_multiple_of) * pad_to_multiple_of
    out = np.full((len(seqs), width), pad_id, dtype=np.int64)
    for row, seq in enumerate(seqs):
        out[row, : seq.shape[0]] = seq
    return torch.from_numpy(out)


class Collator:
    """``list[Example] -> Batch``. A plain class (not a lambda) so it pickles for workers."""

    def __init__(self, pad_id: int = PAD_ID, pad_to_multiple_of: int = 1) -> None:
        self.pad_id = pad_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: Sequence[Example]) -> Batch:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        srcs = [ex.src for ex in examples]
        splits = [teacher_forcing_split(ex.tgt) for ex in examples]
        tgt_ins = [s[0] for s in splits]
        tgt_outs = [s[1] for s in splits]

        src_lengths = torch.tensor([s.shape[0] for s in srcs], dtype=torch.long)
        tgt_lengths = torch.tensor([t.shape[0] for t in tgt_ins], dtype=torch.long)
        return Batch(
            src=pad_sequences(srcs, self.pad_id, self.pad_to_multiple_of),
            tgt_in=pad_sequences(tgt_ins, self.pad_id, self.pad_to_multiple_of),
            tgt_out=pad_sequences(tgt_outs, self.pad_id, self.pad_to_multiple_of),
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            indices=torch.tensor([ex.index for ex in examples], dtype=torch.long),
            num_tgt_tokens=int(tgt_lengths.sum()),
            num_src_tokens=int(src_lengths.sum()),
        )
