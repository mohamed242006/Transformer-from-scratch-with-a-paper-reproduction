"""Metric wrappers vs sacreBLEU directly (ADR-006)."""

from __future__ import annotations

import pytest
import sacrebleu

from aiayn.evaluation import bleu_by_length, corpus_scores, exact_match

HYPS = ["The cat sat on the mat .", "a dog runs", "Hello world"]
REFS = ["The cat sat on the mat .", "A dog is running", "hello world"]


def test_scores_match_sacrebleu_and_carry_signatures() -> None:
    scores = corpus_scores(HYPS, REFS)
    assert scores["bleu"] == pytest.approx(round(sacrebleu.corpus_bleu(HYPS, [REFS]).score, 2))
    assert scores["chrf"] == pytest.approx(round(sacrebleu.corpus_chrf(HYPS, [REFS]).score, 2))
    assert scores["bleu_lowercase"] >= scores["bleu"]  # "Hello"/"hello", "a"/"A" now match
    assert "case:mixed" in scores["bleu_signature"] and "tok:13a" in scores["bleu_signature"]
    assert "case:lc" in scores["bleu_lowercase_signature"]
    assert scores["sentences"] == 3 and len(scores["bleu_precisions"]) == 4


def test_perfect_output() -> None:
    scores = corpus_scores(REFS, REFS)
    assert scores["bleu"] == 100.0 and scores["exact_match"] == 1.0
    assert scores["brevity_penalty"] == 1.0


def test_exact_match_normalizes_whitespace_only() -> None:
    assert exact_match(["a  b", "c"], ["a b", "C"]) == 0.5


def test_bleu_by_length_buckets() -> None:
    buckets = bleu_by_length(HYPS, REFS, [7, 15, 45], edges=(10, 40))
    assert [b["bucket"] for b in buckets] == ["0-9", "10-39", "40+"]
    assert buckets[0]["bleu"] == 100.0 and sum(b["sentences"] for b in buckets) == 3


@pytest.mark.parametrize(("hyps", "refs"), [([], []), (["a"], ["a", "b"])])
def test_invalid_inputs(hyps: list[str], refs: list[str]) -> None:
    with pytest.raises(ValueError):
        corpus_scores(hyps, refs)


# ----------------------------------------------------------------------------- ADR-023 protocol B
def test_fairseq_normalization_matches_prepare_iwslt14() -> None:
    from aiayn.evaluation.metrics import fairseq_normalize

    text = 'It\u2019s a "test" & more, isn\'t it?'
    assert fairseq_normalize(text, "en") == (
        "it &apos;s a &quot; test &quot; &amp; more , isn &apos;t it ?"
    )


def test_fairseq_bleu_is_tokenized_lowercased_corpus_bleu() -> None:
    from sacrebleu.metrics.bleu import BLEU

    from aiayn.evaluation.metrics import corpus_scores, fairseq_bleu, fairseq_normalize

    hyps = ["The Cat sat on the mat.", "It\u2019s raining, isn't it?"]
    refs = ["the cat sat on the mat .".replace(" .", "."), "It's raining, isn't it?"]
    score, signature = fairseq_bleu(hyps, refs, "en")
    manual = BLEU(tokenize="none", smooth_method="none", force=True).corpus_score(
        [fairseq_normalize(h, "en") for h in hyps], [[fairseq_normalize(r, "en") for r in refs]]
    )
    assert score == pytest.approx(manual.score) and score == pytest.approx(100.0)
    assert "fairseq-style" in signature and "sacremoses" in signature
    scores = corpus_scores(hyps, refs, fairseq_lang="en")
    assert scores["bleu_fairseq"] == 100.0 and scores["bleu"] < 100.0  # cased sacreBLEU differs
    assert "bleu_fairseq" not in corpus_scores(hyps, refs)
