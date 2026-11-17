"""Multi-seed statistics (ADR-022): bootstrap BLEU equals sacreBLEU, test calibration on
known cases, exact permutation values, Holm, Spearman, t intervals, p formatting."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from sacrebleu.metrics.bleu import BLEU

from aiayn.evaluation.metrics import corpus_scores
from aiayn.evaluation.statistics import (
    BleuBootstrap,
    compare_bleu,
    format_p,
    holm,
    mean_std,
    permutation_test,
    spearman,
    t_halfwidth,
)

WORDS = [
    "a",
    "man",
    "woman",
    "dog",
    "runs",
    "sits",
    "on",
    "the",
    "red",
    "blue",
    "street",
    "park",
    "ball",
]


def corpus(n: int = 120, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 12))) for _ in range(n)]


def system(refs: list[str], quality: float, seed: int) -> list[str]:
    """Each hypothesis is the reference with probability ``quality``, else truncated."""
    rng = random.Random(seed)
    return [r if rng.random() < quality else " ".join(r.split()[:3]) for r in refs]


REFS = corpus()


def test_bootstrap_bleu_equals_sacrebleu() -> None:
    boot = BleuBootstrap(REFS, n_samples=50)
    for q in (0.2, 0.6, 0.9):
        hyps = system(REFS, q, 1)
        observed, samples = boot.scores(hyps)
        assert observed == pytest.approx(BLEU().corpus_score(hyps, [REFS]).score, abs=1e-9)
        assert round(observed, 2) == corpus_scores(hyps, REFS)["bleu"]
        assert samples.shape == (50,) and np.isfinite(samples).all()


def test_bootstrap_resamples_are_shared_and_reproducible() -> None:
    a, b = BleuBootstrap(REFS, 30, seed=7), BleuBootstrap(REFS, 30, seed=7)
    hyps = system(REFS, 0.5, 2)
    assert np.array_equal(a.scores(hyps)[1], b.scores(hyps)[1])
    assert not np.array_equal(a.scores(hyps)[1], BleuBootstrap(REFS, 30, seed=8).scores(hyps)[1])
    with pytest.raises(ValueError, match="references"):
        a.scores(hyps[:-1])


def test_identical_systems_are_not_significant() -> None:
    boot = BleuBootstrap(REFS, 300)
    seeds = [system(REFS, 0.6, s) for s in range(3)]
    result = compare_bleu(boot, seeds, seeds)
    assert result.delta == 0.0 and result.p_value > 0.5
    assert result.ci_low <= 0.0 <= result.ci_high


def test_large_differences_reach_the_p_floor() -> None:
    boot = BleuBootstrap(REFS, 300)
    good = [system(REFS, 0.95, s) for s in range(5)]
    bad = [system(REFS, 0.2, s + 10) for s in range(3)]  # unequal seed counts are fine
    result = compare_bleu(boot, good, bad)
    assert result.delta > 20 and result.p_value == pytest.approx(1 / 301)
    assert result.ci_low > 0
    assert compare_bleu(boot, bad, good).delta == pytest.approx(-result.delta)


def test_seed_variance_widens_the_interval() -> None:
    """Two-level resampling: noisier seeds -> wider CI than a single-seed system."""
    boot = BleuBootstrap(REFS, 400)
    stable = [system(REFS, 0.6, 1)] * 4
    noisy = [system(REFS, q, i) for i, q in enumerate((0.3, 0.5, 0.7, 0.9))]
    reference = [system(REFS, 0.6, 99)]
    narrow = compare_bleu(boot, stable, reference)
    wide = compare_bleu(boot, noisy, reference)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_permutation_test_exact_values() -> None:
    assert permutation_test([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == pytest.approx(2 / 252)
    assert permutation_test([1, 2, 3], [4, 5, 6, 7, 8]) == pytest.approx(2 / 56)
    assert permutation_test([1.0, 1.0], [1.0, 1.0]) == 1.0
    with pytest.raises(ValueError):
        permutation_test([], [1.0])


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, None, 0.001]) == [pytest.approx(0.5), None, pytest.approx(0.002)]
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # 2 x 0.8 capped at 1, monotone
    assert holm([]) == []


def test_spearman_with_ties_matches_pearson_on_ranks() -> None:
    x = [1.0, 2.0, 2.0, 3.0, 5.0, 4.0]
    y = [10.0, 8.0, 9.0, 7.0, 1.0, 3.0]
    ranks_x = np.array([1, 2.5, 2.5, 4, 6, 5])
    ranks_y = np.array([6, 4, 5, 3, 1, 2])
    expected = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    assert spearman(x, y) == pytest.approx(expected)
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2])


def test_mean_std_and_t_interval() -> None:
    assert mean_std([2.0]) == (2.0, 0.0)
    mean, std = mean_std([1, 2, 3, 4, 5])
    assert mean == 3 and std == pytest.approx(math.sqrt(2.5))
    assert t_halfwidth([1, 2, 3, 4, 5]) == pytest.approx(2.776 * math.sqrt(2.5) / math.sqrt(5))
    assert t_halfwidth([1.0]) == 0.0


def test_format_p() -> None:
    assert format_p(None) == "-"
    assert format_p(1 / 1001, 1000) == "≤0.001"
    assert format_p(0.0123, 1000) == "0.012"
    assert format_p(0.0004) == "<0.001"
