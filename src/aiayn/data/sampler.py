"""Token-budget, length-bucketed batch sampler (paper §5.1; ADR-009).

Batch cost = ``num_sentences * max(max S, max T)``, i.e. padded tokens of the larger side.

Training order (``shuffle=True``) for epoch ``e``:
    rng = default_rng([seed, e]) → permute indices → split into pools of ``pool_size``
    → sort each pool by length → greedy packing within each pool → shuffle batch order.
Evaluation order (``shuffle=False``): global length sort, deterministic.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from torch.utils.data import Sampler

__all__ = ["TokenBudgetBatchSampler"]


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Yield lists of dataset indices whose padded size fits ``max_tokens``.

    Args:
        src_lengths: Source tensor lengths S per example, shape ``[N]``.
        tgt_lengths: Decoder tensor lengths T per example, shape ``[N]``.
        max_tokens: Upper bound on ``len(batch) * max(S, T)`` within a batch.
        max_sentences: Optional upper bound on ``len(batch)``.
        shuffle: Randomize composition and order per epoch (training).
        pool_size: Examples sorted together before packing (only with ``shuffle``).
        seed: Base seed; the effective RNG is ``default_rng([seed, epoch])``.

    Raises:
        ValueError: if any single example exceeds ``max_tokens`` (filter the dataset
            instead of silently dropping data).
    """

    def __init__(
        self,
        src_lengths: npt.ArrayLike,
        tgt_lengths: npt.ArrayLike,
        *,
        max_tokens: int,
        max_sentences: int | None = None,
        shuffle: bool = True,
        pool_size: int = 4096,
        seed: int = 0,
    ) -> None:
        src = np.asarray(src_lengths, dtype=np.int64)
        tgt = np.asarray(tgt_lengths, dtype=np.int64)
        if src.ndim != 1 or src.shape != tgt.shape:
            raise ValueError(f"length arrays must be 1-D and equal-sized: {src.shape}, {tgt.shape}")
        if src.size == 0:
            raise ValueError("cannot sample from an empty dataset")
        if int(np.min(src)) < 1 or int(np.min(tgt)) < 1:
            raise ValueError("all lengths must be >= 1")
        if max_tokens <= 0 or pool_size <= 0 or (max_sentences is not None and max_sentences <= 0):
            raise ValueError("max_tokens, pool_size and max_sentences must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self._sizes = np.maximum(src, tgt)
        oversized = int((self._sizes > max_tokens).sum())
        if oversized:
            raise ValueError(
                f"{oversized} example(s) are longer than max_tokens={max_tokens} "
                f"(longest: {int(self._sizes.max())}); raise max_tokens or filter by max_len"
            )
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.shuffle = shuffle
        self.pool_size = pool_size
        self.seed = seed
        self._epoch = 0
        self._start_batch = 0
        self._cache: tuple[int, list[list[int]]] | None = None

    # ------------------------------------------------------------------ #
    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        """Select the epoch to produce, optionally resuming after ``start_batch`` batches.

        Batches are a pure function of ``(seed, epoch)``, so resuming with the number of
        batches already *consumed by the optimizer* reproduces the uninterrupted run
        exactly (ADR-013). Count consumed batches in the trainer, not here: DataLoader
        prefetching draws batches from this sampler before they are used.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative")
        self._epoch = epoch
        total = len(self._batches())
        if start_batch > total:
            raise ValueError(f"start_batch {start_batch} exceeds {total} batches in epoch {epoch}")
        self._start_batch = start_batch

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def start_batch(self) -> int:
        return self._start_batch

    def state_dict(self) -> dict[str, int | None]:
        """Resumption state plus the settings that must match when it is restored."""
        return {
            "epoch": self._epoch,
            "start_batch": self._start_batch,
            "seed": self.seed,
            "num_examples": int(self._sizes.shape[0]),
            "max_tokens": self.max_tokens,
            "max_sentences": self.max_sentences,
            "pool_size": self.pool_size,
            "shuffle": int(self.shuffle),
        }

    def load_state_dict(self, state: dict[str, int | None]) -> None:
        """Restore ``set_epoch`` state; refuse if the batching settings differ."""
        current = self.state_dict()
        mismatched = {
            k: (state.get(k), v)
            for k, v in current.items()
            if k not in ("epoch", "start_batch") and state.get(k) != v
        }
        if mismatched:
            raise ValueError(
                f"sampler settings differ from checkpoint (saved, current): {mismatched}"
            )
        epoch, start = state["epoch"], state["start_batch"]
        if epoch is None or start is None:
            raise ValueError("state is missing epoch/start_batch")
        self.set_epoch(epoch, start)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()[self._start_batch :]

    def __len__(self) -> int:
        return len(self._batches()) - self._start_batch

    def padding_efficiency(self) -> float:
        """Real tokens / padded tokens (larger side) over this epoch's batches, in (0, 1]."""
        real = padded = 0
        for batch in self._batches():
            sizes = self._sizes[batch]
            real += int(sizes.sum())
            padded += len(batch) * int(sizes.max())
        return real / padded

    # ------------------------------------------------------------------ #
    def _batches(self) -> list[list[int]]:
        if self._cache is not None and self._cache[0] == self._epoch:
            return self._cache[1]
        n = self._sizes.shape[0]
        if self.shuffle:
            rng = np.random.default_rng([self.seed, self._epoch])
            order = rng.permutation(n)
            pools = [order[i : i + self.pool_size] for i in range(0, n, self.pool_size)]
        else:
            rng = None
            pools = [np.arange(n)]

        batches: list[list[int]] = []
        for pool in pools:
            ordered = pool[np.argsort(self._sizes[pool], kind="stable")]
            batches.extend(self._pack(ordered))

        if rng is not None:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        self._cache = (self._epoch, batches)
        return batches

    def _pack(self, ordered: npt.NDArray[np.int64]) -> list[list[int]]:
        """Greedy packing of length-sorted indices under the token/sentence budgets."""
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for idx in ordered.tolist():
            size = int(self._sizes[idx])
            new_max = max(current_max, size)
            over_tokens = (len(current) + 1) * new_max > self.max_tokens
            over_sents = self.max_sentences is not None and len(current) + 1 > self.max_sentences
            if current and (over_tokens or over_sents):
                batches.append(current)
                current, new_max = [], size
            current.append(idx)
            current_max = new_max
        if current:
            batches.append(current)
        return batches
