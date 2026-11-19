"""The encoder-decoder interface shared by the Transformer and the RNN baselines (ADR-021).

Training, decoding, the translator and evaluation depend only on this interface, so every
model goes through *identical* batching, loss, checkpointing, beam search and scoring.

Contract (B batch, S source length, T decoder length, V vocabulary):

* ``encode(src [B,S]) -> (memory, src_mask, maps)``. ``memory`` is opaque to callers
  except that it supports ``memory.repeat_interleave(k, dim=0)`` (beam expansion):
  a tensor for the Transformer, an ``RNNMemory`` for the baselines. ``src_mask`` is a
  boolean tensor with the batch in dimension 0 (True = may attend).
* ``decode(tgt_in [B,T], memory, src_mask) -> (hidden [B,T,*], self maps, cross maps)``:
  teacher-forced and causal (position t depends only on ``tgt_in[:, :t+1]``), so
  decoding by recomputing the prefix is exact for every model.
* ``generator(hidden) -> logits [B,T,V]``: raw logits, no softmax.
* ``max_positions``: the decode-length cap (the Transformer's positional table size).

Incremental decoding (ADR-023 §6), used by beam/greedy search when available:

* ``init_state(memory, src_mask) -> state``: after beam expansion of the memory;
* ``decode_step(tokens [N], state) -> (hidden [N, *], state)``: feed the newest token;
* ``reorder_state(state, rows [N]) -> state``: beam reselection of batch rows.

Each step must equal ``decode`` over the whole prefix at the last position (tested).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["EncoderDecoder"]


class EncoderDecoder(nn.Module):
    """Abstract base: subclasses set ``vocab_size``/``pad_id`` and implement the methods."""

    vocab_size: int
    pad_id: int
    #: Expensive invariant checks on every call (Transformer: masks, token ranges, finite
    #: logits). Models without such checks ignore it.
    debug_checks: bool = False

    @property
    def max_positions(self) -> int:
        raise NotImplementedError

    def encode(
        self, src: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[Any, torch.Tensor, tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: Any,
        src_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    def generator(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    #: True when ``init_state``/``decode_step``/``reorder_state`` are implemented.
    supports_incremental: bool = False

    def init_state(self, memory: Any, src_mask: torch.Tensor) -> Any:
        raise NotImplementedError

    def decode_step(self, tokens: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def reorder_state(self, state: Any, rows: torch.Tensor) -> Any:
        raise NotImplementedError

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits ``[B, T, V]``."""
        memory, src_mask, _ = self.encode(src)
        hidden, _, _ = self.decode(tgt_in, memory, src_mask)
        return self.generator(hidden)
