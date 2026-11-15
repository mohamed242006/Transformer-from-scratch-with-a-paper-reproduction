# ADR-006: Evaluation protocol: sacreBLEU primary, reference-protocol BLEU secondary

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [ADR-002](ADR-002-iwslt14-before-wmt14.md), [ADR-005](ADR-005-sentencepiece-tokenization.md), [paper_notes.md §7.1](../paper_notes.md), spec §8

## Context
BLEU depends heavily on tokenization, casing and the reference format (Post, 2018). The paper does not state its BLEU tool. The tensor2tensor EN-DE numbers are tokenized BLEU with compound splitting. Common IWSLT14 reference numbers are **lowercased, tokenized** BLEU. Mixing protocols is the most common way reproductions mislead people. ADR-002 promised this ADR in Week 2, before any metric code exists, so the protocol is fixed *before* we see any results.

## Decision
1. **Primary metric:** corpus-level **sacreBLEU** on **detokenized, cased** hypotheses against the **raw** references, with default settings (`13a` tokenizer). The sacreBLEU **signature** string is stored with every reported number.
2. **Secondary metric (only for comparing with published numbers):** BLEU under the *reference's* protocol, computed from the **same hypotheses**:
   - IWSLT14 (fairseq-style): lowercase both sides; sacreBLEU with `--lowercase` and the `13a` tokenizer as the closest standard equivalent. The residual mismatch with Moses-tokenized multi-bleu is acknowledged (±~0.5 BLEU).
   - WMT14 EN-DE (paper): tokenized BLEU with compound splitting, implemented as documented by tensor2tensor, reported **only** next to the paper's 27.3 / 28.4.
3. **chrF** (sacreBLEU, default settings) is reported alongside BLEU.
4. **Significance:** paired bootstrap resampling (1000 samples, sacreBLEU implementation) for every "A beats B" claim.
5. **Splits:** model selection and checkpoint choice use **validation only**. The test set is decoded once per final model.
6. **Perplexity** is exp(mean unsmoothed NLL per non-pad target token) and is compared only between models sharing a tokenizer.

## Consequences
**Positive:** results are comparable across papers and stay honest; the protocol is fixed before results exist. **Negative:** two BLEU columns to explain; the secondary protocol is approximate. **Risk:** a hypothesis/reference ordering bug (the batch sampler reorders sentences) → mitigated by carrying dataset `indices` through every batch (ADR-009) and a test that reassembled hypotheses are in corpus order.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Only sacreBLEU | Cannot be compared with IWSLT/WMT literature numbers |
| Only reference-protocol BLEU | Not reproducible; hides tokenization effects |
| Home-made BLEU | Not comparable; allowed only as an exercise validated against sacreBLEU |
| COMET as primary | Model-dependent, heavy; kept as an extension |

## Compliance
Evaluation code (Week 6) writes `{bleu, signature, chrf, protocol}` to `metrics.json`. Review checklist: every BLEU in a document names its protocol.
