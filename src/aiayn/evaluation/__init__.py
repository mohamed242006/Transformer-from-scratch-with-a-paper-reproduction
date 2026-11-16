"""Metric wrappers (sacreBLEU, chrF), exact match and length-bucket analysis.

Works on strings only; never imports models (docs/REPOSITORY.md §3).

Owner: Evaluation owner. Contract: docs/REPOSITORY.md#srcaiaynevaluation.
"""

from aiayn.evaluation.metrics import bleu_by_length, corpus_scores, exact_match

__all__ = ["bleu_by_length", "corpus_scores", "exact_match"]
