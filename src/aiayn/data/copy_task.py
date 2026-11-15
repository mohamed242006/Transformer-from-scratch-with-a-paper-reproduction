"""Synthetic copy / reverse task: the first correctness check for any seq2seq model.

Each example is a random sequence of content tokens. The target is the same sequence
(``copy``) or its reverse (``reverse``). A correct Transformer reaches ~100% sequence
accuracy, so any failure points at the model or masking, never at the data.

Examples are generated lazily and deterministically from ``(seed, stream, index)``.
The result does not depend on access order or on which DataLoader worker draws it,
and the dataset pickles in O(1) apart from its length array.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from torch.utils.data import Dataset

from aiayn.data.example import Example, frame_source, frame_target
from aiayn.data.special_tokens import NUM_SPECIAL_TOKENS

__all__ = ["CopyTaskDataset"]


class CopyTaskDataset(Dataset[Example]):
    """Random sequences over IDs ``[NUM_SPECIAL_TOKENS, vocab_size)``.

    Shapes per example (L = content length, uniform in ``[min_len, max_len]``):
        src: ``[L + 1]`` (``x EOS``), tgt: ``[L + 2]`` (``BOS y EOS``).
    Dataset lengths: ``src_lengths = tgt_lengths = L + 1`` (tensor lengths S and T).
    """

    def __init__(
        self,
        *,
        num_examples: int,
        vocab_size: int,
        min_len: int,
        max_len: int,
        seed: int,
        stream: int = 0,
        mode: Literal["copy", "reverse"] = "copy",
    ) -> None:
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
        if vocab_size <= NUM_SPECIAL_TOKENS:
            raise ValueError(f"vocab_size must exceed {NUM_SPECIAL_TOKENS} special tokens")
        if not 1 <= min_len <= max_len:
            raise ValueError(f"need 1 <= min_len <= max_len, got {min_len}, {max_len}")
        if seed < 0 or stream < 0:
            raise ValueError("seed and stream must be non-negative")
        if mode not in ("copy", "reverse"):
            raise ValueError(f"unknown mode {mode!r}")
        self.num_examples = num_examples
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.stream = stream
        self.mode = mode
        rng = np.random.default_rng([seed, stream, 0])
        self._content_lengths = rng.integers(min_len, max_len + 1, size=num_examples)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> Example:
        if not 0 <= index < self.num_examples:
            raise IndexError(f"index {index} out of range for {self.num_examples} examples")
        rng = np.random.default_rng([self.seed, self.stream, 1, index])
        length = int(self._content_lengths[index])
        content = rng.integers(NUM_SPECIAL_TOKENS, self.vocab_size, size=length)
        target = content if self.mode == "copy" else content[::-1]
        return Example(index=index, src=frame_source(content), tgt=frame_target(target))

    @property
    def src_lengths(self) -> np.ndarray:
        """Source tensor lengths S (content + EOS), shape ``[N]``."""
        return self._content_lengths + 1

    @property
    def tgt_lengths(self) -> np.ndarray:
        """Decoder tensor lengths T (content + BOS or EOS), shape ``[N]``."""
        return self._content_lengths + 1
