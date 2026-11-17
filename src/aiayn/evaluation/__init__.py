"""Metric wrappers (sacreBLEU, chrF), exact match, length-bucket analysis and paired
bootstrap significance.

Works on strings only; never imports models (docs/REPOSITORY.md §3).

Owner: Evaluation owner. Contract: docs/REPOSITORY.md#srcaiaynevaluation.
"""

from aiayn.evaluation.metrics import bleu_by_length, corpus_scores, exact_match
from aiayn.evaluation.significance import PairedResult, paired_bootstrap

__all__ = [
    "PairedResult",
    "bleu_by_length",
    "corpus_scores",
    "exact_match",
    "paired_bootstrap",
]
