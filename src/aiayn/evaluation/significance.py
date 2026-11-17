"""Paired bootstrap resampling (Koehn, 2004) via sacreBLEU's ``PairedTest`` (ADR-006, ADR-021).

The first system is the reference ("baseline" in sacreBLEU's terms). For every other
system sacreBLEU reports the bootstrap mean, the 95% confidence half-width and the
p-value of the null hypothesis that it is no different from the reference.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.significance import PairedTest

__all__ = ["PairedResult", "paired_bootstrap"]


@dataclass(frozen=True)
class PairedResult:
    name: str
    bleu: float
    bleu_mean: float
    bleu_ci: float  # 95% half-width
    bleu_p: float | None  # None for the reference system
    chrf: float
    chrf_mean: float
    chrf_ci: float
    chrf_p: float | None


def paired_bootstrap(
    systems: Sequence[tuple[str, Sequence[str]]],
    references: Sequence[str],
    *,
    n_samples: int = 1000,
) -> tuple[list[PairedResult], str]:
    """Return one ``PairedResult`` per system (reference first) and the BLEU signature."""
    if len(systems) < 2:
        raise ValueError("need a reference system and at least one other system")
    for name, hyps in systems:
        if len(hyps) != len(references):
            raise ValueError(f"{name}: {len(hyps)} hypotheses vs {len(references)} references")
    sacre_logger = logging.getLogger("sacrebleu")
    level = sacre_logger.level
    sacre_logger.setLevel(logging.ERROR)  # silence "parallel tests not supported" notices
    try:
        test = PairedTest(
            [(name, list(hyps)) for name, hyps in systems],
            {"BLEU": BLEU(), "chrF": CHRF()},
            [list(references)],
            test_type="bs",
            n_samples=n_samples,
        )
        signatures, scores = test()
    finally:
        sacre_logger.setLevel(level)
    chrf_key = next(k for k in scores if k.startswith("chrF"))
    results = []
    for i, (name, _) in enumerate(systems):
        b, c = scores["BLEU"][i], scores[chrf_key][i]
        assert not isinstance(b, str) and not isinstance(c, str)
        results.append(
            PairedResult(
                name=name,
                bleu=float(b.score),
                bleu_mean=float(b.mean or 0.0),
                bleu_ci=float(b.ci or 0.0),
                bleu_p=None if b.p_value is None else float(b.p_value),
                chrf=float(c.score),
                chrf_mean=float(c.mean or 0.0),
                chrf_ci=float(c.ci or 0.0),
                chrf_p=None if c.p_value is None else float(c.p_value),
            )
        )
    return results, str(signatures["BLEU"])
