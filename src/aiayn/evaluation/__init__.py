"""Metric wrappers (sacreBLEU, chrF), exact match, length-bucket analysis, paired
bootstrap significance and multi-seed statistics.

Works on strings only; never imports models (docs/REPOSITORY.md §3).

Owner: Evaluation owner. Contract: docs/REPOSITORY.md#srcaiaynevaluation.
"""

from aiayn.evaluation.metrics import bleu_by_length, corpus_scores, exact_match
from aiayn.evaluation.significance import PairedResult, paired_bootstrap
from aiayn.evaluation.statistics import (
    BleuBootstrap,
    BootstrapComparison,
    compare_bleu,
    format_p,
    holm,
    mean_std,
    permutation_test,
    spearman,
    t_halfwidth,
)

__all__ = [
    "BleuBootstrap",
    "BootstrapComparison",
    "PairedResult",
    "bleu_by_length",
    "compare_bleu",
    "corpus_scores",
    "exact_match",
    "format_p",
    "holm",
    "mean_std",
    "paired_bootstrap",
    "permutation_test",
    "spearman",
    "t_halfwidth",
]
