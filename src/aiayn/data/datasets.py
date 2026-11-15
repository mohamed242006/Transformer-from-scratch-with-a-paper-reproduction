"""Parallel-text datasets: reading, tokenization, framing, filtering, flat storage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from aiayn.data.example import TOKEN_DTYPE, Example, frame_source, frame_target
from aiayn.data.tokenizer import Tokenizer

__all__ = [
    "FilterStats",
    "SequencePairDataset",
    "TranslationDataset",
    "read_lines",
    "read_parallel_text",
]

_ENCODE_CHUNK = 10_000


class SequencePairDataset(Protocol):
    """What samplers and loaders need from a dataset (implemented by both datasets)."""

    #: Size of the ID space the model's embedding must cover (specials included).
    vocab_size: int

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Example: ...

    @property
    def src_lengths(self) -> np.ndarray: ...

    @property
    def tgt_lengths(self) -> np.ndarray: ...


def read_lines(path: str | Path) -> list[str]:
    """Read UTF-8 text, splitting **only** on ``\\n`` (``\\r\\n`` tolerated).

    ``str.splitlines`` would also split on U+2028, U+0085 and similar characters, which
    occur in web-crawled corpora and would silently misalign parallel files.
    """
    with Path(path).open(encoding="utf-8", newline="") as fh:
        text = fh.read()
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        return []
    return [line.removesuffix("\r") for line in text.split("\n")]


def read_parallel_text(src_path: str | Path, tgt_path: str | Path) -> tuple[list[str], list[str]]:
    """Read aligned source/target files; their line counts must match."""
    src, tgt = read_lines(src_path), read_lines(tgt_path)
    if len(src) != len(tgt):
        raise ValueError(
            f"parallel files are misaligned: {src_path} has {len(src)} lines, "
            f"{tgt_path} has {len(tgt)}"
        )
    return src, tgt


@dataclass(frozen=True)
class FilterStats:
    """Counts from length filtering; reported by scripts and stored with runs."""

    total: int
    kept: int
    too_short: int
    too_long: int
    bad_ratio: int

    @property
    def dropped(self) -> int:
        return self.total - self.kept


def _flatten(seqs: Sequence[np.ndarray]) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.concatenate(seqs).astype(TOKEN_DTYPE) if seqs else np.zeros(0, TOKEN_DTYPE)
    return flat, offsets


class TranslationDataset(Dataset[Example]):
    """Tokenized, framed and length-filtered parallel corpus.

    Sequences are stored as one flat ``int32`` array plus offsets per side. This keeps
    memory low (≈ 4 bytes/token) and makes pickling to DataLoader workers fast.

    Lengths exposed to samplers are tensor lengths: ``S = n + 1`` and ``T = m + 1``.
    Filtering keeps a pair iff ``n, m >= min_len``, ``S, T <= max_len`` and, if set,
    ``max(n, m) / min(n, m) <= max_len_ratio``.
    """

    def __init__(
        self,
        src_texts: Sequence[str],
        tgt_texts: Sequence[str],
        tokenizer: Tokenizer,
        *,
        max_len: int,
        min_len: int = 1,
        max_len_ratio: float | None = None,
    ) -> None:
        if len(src_texts) != len(tgt_texts):
            raise ValueError(f"{len(src_texts)} source vs {len(tgt_texts)} target sentences")
        if not 1 <= min_len < max_len:
            raise ValueError(f"need 1 <= min_len < max_len, got {min_len}, {max_len}")

        src_kept: list[np.ndarray] = []
        tgt_kept: list[np.ndarray] = []
        line_numbers: list[int] = []
        too_short = too_long = bad_ratio = 0

        for start in range(0, len(src_texts), _ENCODE_CHUNK):
            src_ids = tokenizer.encode_batch(src_texts[start : start + _ENCODE_CHUNK])
            tgt_ids = tokenizer.encode_batch(tgt_texts[start : start + _ENCODE_CHUNK])
            for offset, (s, t) in enumerate(zip(src_ids, tgt_ids, strict=True)):
                n, m = len(s), len(t)
                if n < min_len or m < min_len:
                    too_short += 1
                elif n + 1 > max_len or m + 1 > max_len:
                    too_long += 1
                elif max_len_ratio is not None and max(n, m) / min(n, m) > max_len_ratio:
                    bad_ratio += 1
                else:
                    src_kept.append(frame_source(s))
                    tgt_kept.append(frame_target(t))
                    line_numbers.append(start + offset)

        if not line_numbers:
            raise ValueError("no sentence pairs survived filtering")

        self._src, self._src_offsets = _flatten(src_kept)
        self._tgt, self._tgt_offsets = _flatten(tgt_kept)
        self._line_numbers = np.asarray(line_numbers, dtype=np.int64)
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer_fingerprint = tokenizer.fingerprint
        self.filter_stats = FilterStats(
            total=len(src_texts),
            kept=len(line_numbers),
            too_short=too_short,
            too_long=too_long,
            bad_ratio=bad_ratio,
        )

    def __len__(self) -> int:
        return int(self._line_numbers.shape[0])

    def __getitem__(self, index: int) -> Example:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} examples")
        s0, s1 = self._src_offsets[index], self._src_offsets[index + 1]
        t0, t1 = self._tgt_offsets[index], self._tgt_offsets[index + 1]
        return Example(index=index, src=self._src[s0:s1], tgt=self._tgt[t0:t1])

    @property
    def src_lengths(self) -> np.ndarray:
        """Source tensor lengths S, shape ``[N]``."""
        return np.diff(self._src_offsets)

    @property
    def tgt_lengths(self) -> np.ndarray:
        """Decoder tensor lengths T = (framed target length) - 1, shape ``[N]``."""
        return np.diff(self._tgt_offsets) - 1

    def line_number(self, index: int) -> int:
        """Original (0-based) line of example ``index`` in the source files."""
        return int(self._line_numbers[index])

    @property
    def num_tokens(self) -> tuple[int, int]:
        """Total framed (source, target) tokens, for logging."""
        return int(self._src.shape[0]), int(self._tgt.shape[0])
