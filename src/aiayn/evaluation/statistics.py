"""Statistics for multi-seed comparisons (ADR-022).

* ``mean_std`` / ``t_halfwidth``: seed-level summaries (Student t, 95%).
* ``permutation_test``: exact two-sided test on seed-level scores (training variance only).
* ``BleuBootstrap`` + ``compare_bleu``: the primary test. A **two-level paired bootstrap**
  resamples test sentences (shared by both systems, so the comparison is paired) *and*
  each system's seeds, so the confidence interval and p-value reflect test-set sampling
  and training variance together. Corpus BLEU on each resample is computed from
  sacreBLEU's own sufficient statistics, so it equals sacreBLEU exactly (tested).
* ``holm``: Holm-Bonferroni adjustment within a family of comparisons.
* ``spearman``: rank correlation (average ranks for ties).

The p-value of a bootstrap test is ``(1 + #{|Δ*_b − mean Δ*| ≥ |Δ|}) / (B + 1)``, so the
smallest reportable value is ``1 / (B + 1)``; ``format_p`` prints it as ``≤0.001``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sacrebleu.metrics.bleu import BLEU

__all__ = [
    "BleuBootstrap",
    "BootstrapComparison",
    "compare_bleu",
    "format_p",
    "holm",
    "mean_std",
    "permutation_test",
    "spearman",
    "t_halfwidth",
]

# Two-sided 95% Student t quantiles, df = 1..30.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    if not values:
        raise ValueError("no values")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def t_halfwidth(values: Sequence[float]) -> float:
    """Half-width of the 95% Student t confidence interval of the mean (0 for n < 2)."""
    n = len(values)
    if n < 2:
        return 0.0
    quantile = _T975[n - 2] if n - 1 <= len(_T975) else 1.96
    return quantile * mean_std(values)[1] / math.sqrt(n)


def permutation_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided permutation test of ``mean(a) − mean(b)`` (all relabelings)."""
    if not a or not b:
        raise ValueError("both samples need at least one value")
    pooled = [*a, *b]
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    total = sum(pooled)
    count = n = 0
    for chosen in itertools.combinations(range(len(pooled)), len(a)):
        sum_a = sum(pooled[i] for i in chosen)
        diff = abs(sum_a / len(a) - (total - sum_a) / len(b))
        count += diff >= observed - 1e-9
        n += 1
    return count / n


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (``None`` entries are skipped and kept)."""
    present = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    m = len(present)
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def _ranks(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = np.asarray(values)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("spearman needs two equally long samples of at least 3 values")
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt(float((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom else 0.0


def format_p(p: float | None, n_samples: int | None = None) -> str:
    """``-`` for None; ``≤0.001`` at the bootstrap floor ``1/(B+1)``; else 3 decimals."""
    if p is None:
        return "-"
    if n_samples is not None and p <= 1.0 / (n_samples + 1) + 1e-12:
        return f"≤{1.0 / (n_samples + 1):.3f}"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


class BleuBootstrap:
    """Shared sentence resamples for one reference set; BLEU per resample per system.

    All systems compared with one instance see the *same* resamples (paired test).
    """

    def __init__(self, references: Sequence[str], n_samples: int = 1000, seed: int = 12345):
        if not references:
            raise ValueError("no references")
        self.references = list(references)
        self.n_samples = n_samples
        self.seed = seed
        self._bleu = BLEU()
        n = len(self.references)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(n_samples, n))
        self._weights = np.zeros((n_samples, n), dtype=np.int64)
        np.add.at(self._weights, (np.arange(n_samples)[:, None], draws), 1)
        self._cache: dict[tuple[str, ...], tuple[float, np.ndarray]] = {}

    def _score(self, stats: Sequence[int]) -> float:
        return float(self._bleu._compute_score_from_stats(list(stats)).score)

    def scores(self, hypotheses: Sequence[str]) -> tuple[float, np.ndarray]:
        """``(corpus BLEU, BLEU on each of the B resamples)``."""
        key = tuple(hypotheses)
        if key not in self._cache:
            if len(hypotheses) != len(self.references):
                raise ValueError(
                    f"{len(hypotheses)} hypotheses vs {len(self.references)} references"
                )
            stats = np.asarray(
                self._bleu._extract_corpus_statistics(list(hypotheses), [self.references]),
                dtype=np.int64,
            )  # [N, 2 + 2·4]
            observed = self._score(stats.sum(axis=0).tolist())
            resampled = self._weights @ stats  # [B, stats]
            samples = np.array([self._score(row.tolist()) for row in resampled])
            self._cache[key] = (observed, samples)
        return self._cache[key]


@dataclass(frozen=True)
class BootstrapComparison:
    delta: float  # mean-over-seeds BLEU(a) − BLEU(b)
    ci_low: float
    ci_high: float
    p_value: float
    n_samples: int


def compare_bleu(
    bootstrap: BleuBootstrap,
    system_a: Sequence[Sequence[str]],
    system_b: Sequence[Sequence[str]],
    *,
    seed: int = 0,
) -> BootstrapComparison:
    """Two-level paired bootstrap of ``mean BLEU(a) − mean BLEU(b)``.

    ``system_a``/``system_b`` hold one hypothesis list per seed (any number of seeds each).
    Per resample: the shared sentence resample, then seeds drawn with replacement within
    each system.
    """
    if not system_a or not system_b:
        raise ValueError("each system needs at least one seed")
    rng = np.random.default_rng(seed)
    b_count = bootstrap.n_samples
    columns = np.arange(b_count)[:, None]

    def resampled_means(system: Sequence[Sequence[str]]) -> tuple[float, np.ndarray]:
        pairs = [bootstrap.scores(h) for h in system]
        observed = float(np.mean([p[0] for p in pairs]))
        table = np.stack([p[1] for p in pairs], axis=1)  # [B, seeds]
        picks = rng.integers(0, len(system), size=(b_count, len(system)))
        return observed, table[columns, picks].mean(axis=1)

    obs_a, samples_a = resampled_means(system_a)
    obs_b, samples_b = resampled_means(system_b)
    delta = obs_a - obs_b
    deltas = samples_a - samples_b
    centered = np.abs(deltas - deltas.mean())
    p = (1 + int((centered >= abs(delta) - 1e-12).sum())) / (b_count + 1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return BootstrapComparison(delta, float(low), float(high), p, b_count)
