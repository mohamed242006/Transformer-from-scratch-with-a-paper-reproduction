"""Corpus metrics (ADR-006): sacreBLEU (primary, cased), lowercased BLEU (secondary, for
comparison with lowercased literature numbers), chrF, exact match, BLEU by length, and the
fairseq-style tokenized lowercased BLEU used by IWSLT14 papers (ADR-023, protocol B).

This module is the only place that imports ``sacrebleu`` (enforced by the layering test).
It works on strings only and never imports models.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

__all__ = ["bleu_by_length", "corpus_scores", "exact_match", "fairseq_bleu", "fairseq_normalize"]

DEFAULT_EDGES: tuple[int, ...] = (10, 20, 30, 40)


def _check(hypotheses: Sequence[str], references: Sequence[str]) -> None:
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    if not hypotheses:
        raise ValueError("cannot score an empty corpus")


def exact_match(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Fraction of hypotheses identical to their reference (whitespace-normalized)."""
    _check(hypotheses, references)
    same = sum(
        " ".join(h.split()) == " ".join(r.split())
        for h, r in zip(hypotheses, references, strict=True)
    )
    return same / len(hypotheses)


FAIRSEQ_SIGNATURE = (
    "fairseq-style|moses:sacremoses-{version}|escape|lc|apostrophe:U+2019->'|"
    "sacrebleu tok:none|smooth:none|nrefs:1"
)


@lru_cache(maxsize=4)
def _moses(lang: str) -> Any:
    from sacremoses import MosesTokenizer

    return MosesTokenizer(lang=lang)


def fairseq_normalize(text: str, lang: str) -> str:
    """fairseq's IWSLT14 reference processing (``prepare-iwslt14.sh``): U+2019 -> ', Moses
    ``tokenizer.perl -l <lang>`` (escaping on), lowercase. Applied to hypotheses too."""
    tokenized = _moses(lang).tokenize(text.replace("\u2019", "'"), escape=True, return_str=True)
    return str(tokenized).lower()


def fairseq_bleu(
    hypotheses: Sequence[str], references: Sequence[str], lang: str
) -> tuple[float, str]:
    """Tokenized lowercased corpus BLEU as ``multi-bleu.perl``/``fairseq-score`` compute it.

    Returns ``(score, signature)``. Not comparable to sacreBLEU numbers (ADR-006, ADR-023).
    """
    _check(hypotheses, references)
    from importlib.metadata import version

    hyps = [fairseq_normalize(h, lang) for h in hypotheses]
    refs = [fairseq_normalize(r, lang) for r in references]
    score = BLEU(tokenize="none", smooth_method="none", force=True).corpus_score(hyps, [refs])
    return float(score.score), FAIRSEQ_SIGNATURE.format(version=version("sacremoses"))


def corpus_scores(
    hypotheses: Sequence[str], references: Sequence[str], *, fairseq_lang: str | None = None
) -> dict[str, Any]:
    """Corpus-level scores with sacreBLEU signatures (detokenized text in, ADR-006).

    With ``fairseq_lang`` (the target language), also the fairseq-style tokenized
    lowercased BLEU (``bleu_fairseq``, ADR-023 protocol B).
    """
    _check(hypotheses, references)
    hyps, refs = list(hypotheses), [list(references)]
    bleu_metric = BLEU()
    bleu = bleu_metric.corpus_score(hyps, refs)
    lower_metric = BLEU(lowercase=True)
    bleu_lower = lower_metric.corpus_score(hyps, refs)
    chrf_metric = CHRF()
    chrf = chrf_metric.corpus_score(hyps, refs)
    result: dict[str, Any] = {
        "bleu": round(bleu.score, 2),
        "bleu_signature": str(bleu_metric.get_signature()),
        "bleu_precisions": [round(p, 1) for p in bleu.precisions],
        "brevity_penalty": round(bleu.bp, 4),
        "length_ratio": round(bleu.sys_len / max(bleu.ref_len, 1), 4),
        "bleu_lowercase": round(bleu_lower.score, 2),
        "bleu_lowercase_signature": str(lower_metric.get_signature()),
        "chrf": round(chrf.score, 2),
        "chrf_signature": str(chrf_metric.get_signature()),
        "exact_match": round(exact_match(hyps, list(references)), 4),
        "sentences": len(hyps),
    }
    if fairseq_lang is not None:
        score, signature = fairseq_bleu(hyps, list(references), fairseq_lang)
        result["bleu_fairseq"] = round(score, 2)
        result["bleu_fairseq_signature"] = signature
    return result


def bleu_by_length(
    hypotheses: Sequence[str],
    references: Sequence[str],
    source_lengths: Sequence[int],
    edges: Sequence[int] = DEFAULT_EDGES,
) -> list[dict[str, Any]]:
    """BLEU per source-length bucket (spec §7.4): ``[0, e1), [e1, e2), …, [e_last, ∞)``."""
    _check(hypotheses, references)
    if len(source_lengths) != len(hypotheses):
        raise ValueError("source_lengths must align with hypotheses")
    lows = [0, *edges]
    highs: list[int | None] = [*edges, None]
    buckets: list[dict[str, Any]] = []
    for low, high in zip(lows, highs, strict=True):
        idx = [i for i, n in enumerate(source_lengths) if n >= low and (high is None or n < high)]
        if not idx:
            continue
        hyps = [hypotheses[i] for i in idx]
        refs = [references[i] for i in idx]
        score = BLEU().corpus_score(hyps, [refs]).score
        label = f"{low}-{high - 1}" if high is not None else f"{low}+"
        buckets.append({"bucket": label, "sentences": len(idx), "bleu": round(score, 2)})
    return buckets
