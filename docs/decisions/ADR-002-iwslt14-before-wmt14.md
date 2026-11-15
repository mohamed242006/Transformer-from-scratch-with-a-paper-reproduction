# ADR-002: Use IWSLT14 DE→EN as the primary benchmark before WMT14 EN→DE

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** spec §5 and §6.2, [paper_notes.md §6–7](../paper_notes.md), planned ADR-006 (evaluation protocol)

## Context

The paper's headline benchmark is **WMT14 EN→DE** (about 4.5M sentence pairs, ~37k shared BPE vocabulary, ~25k+25k tokens per batch, 100k steps on 8×P100 ≈ 96 GPU-hours for the base model; paper §5). The project's compute budget is **one GPU**, consumer-grade or briefly rented, and the plan needs **many** runs: baselines, ablations, and ≥ 3 seeds for headline claims.

Rough budget arithmetic:
- A WMT14 base run on one modern GPU without fused kernels is estimated at **1–3 days**. Twenty such runs (baselines × seeds × ablations) is not feasible.
- An IWSLT14 run (~160k pairs) with an IWSLT-sized model is about **6–24 hours**. A Multi30k run (~29k pairs) is **under 1 hour**.

We still need a benchmark that is (a) the **same task** as the paper (encoder-decoder MT measured by BLEU), (b) **standard** enough to have published Transformer reference numbers, and (c) **affordable**.

There are complications we must not hide:
1. **Protocol comparability.** Commonly cited IWSLT14 DE→EN Transformer numbers (about 34–35 BLEU) come from fairseq-style preprocessing, which **lowercases** and Moses-tokenizes the text, and computes **tokenized** BLEU. Those numbers are **not comparable** with cased, detokenized sacreBLEU.
2. **Data availability.** The original IWSLT14 distribution (WIT3) has been intermittently unavailable, so reproductions depend on mirrors.
3. **Direction.** IWSLT14 is conventionally evaluated **DE→EN**, while the paper's task is EN→DE.

## Decision

1. **Dataset ladder.** Synthetic copy task (correctness) → **Multi30k EN-DE** (development) → **IWSLT14 DE→EN (primary benchmark)** → WMT14 EN→DE (optional, time-boxed stretch).
2. **All headline results, baselines and ablations** are produced on IWSLT14 DE→EN, with Multi30k as the cheap fallback for ablations if compute runs short (and declared as such).
3. **The data source is pinned.** The exact source URL or mirror, the file list and **SHA-256 checksums** are recorded in `configs/data/` and verified by the preparation script. If IWSLT14 cannot be obtained reproducibly, **IWSLT2017 EN-DE** (available through maintained dataset hubs) is the documented fallback, through a superseding ADR.
4. **Dual BLEU reporting** (formalized in ADR-006, Week 2):
   - **Primary:** sacreBLEU on detokenized, cased output, with the full signature.
   - **Secondary, for comparison with the literature:** BLEU computed under the reference protocol (lowercased, tokenized), produced from the *same* hypotheses.
   - Every comparison to a published number states which protocol it uses.
5. **WMT14 is attempted only after** all core deliverables are complete (roadmap Week 12+), with a cloud-cost cap fixed before starting.

## Consequences

### Positive
- About 20–50× cheaper per run than WMT14, which makes multiple seeds and full ablation tables feasible, so our claims can be **statistically supported**, unlike the single-run Table 3 in the paper.
- Same task family and architecture as the paper, so every component (encoder, cross-attention, beam search, BLEU) is exercised.
- Published IWSLT reference numbers give an external sanity check.

### Negative / costs
- We **cannot claim** to reproduce 27.3/28.4 BLEU. We reproduce the *method* and the *relative findings*. All public material must say so explicitly.
- Small data needs **regularization different from the paper** (dropout 0.3, d_ff = 1024, h = 4), which is itself a deviation to document.
- Some effects that appear at WMT scale (for example "bigger is better", Table 3 (C)) may **not** appear on small data, and may even reverse because of overfitting. That is a result to report, not to hide.
- DE→EN instead of EN→DE: German morphology is harder to *generate*, so DE→EN BLEU is typically higher. This does not affect the comparisons inside the project.

### Risks and mitigations
| Risk | Mitigation |
|---|---|
| Comparing our sacreBLEU against a lowercased, tokenized reference number (the most common reproduction mistake) | Dual reporting, protocol stated next to every number |
| The IWSLT14 mirror disappears | Checksums + documented IWSLT2017 fallback; raw data never committed (licence: TED talks are CC BY-NC-ND) |
| Ablation results don't transfer to WMT scale | State the scale caveat; optional WMT14 run checks the headline configuration |
| Tokenizer differences (SentencePiece vs Moses+BPE, ADR-005) shift BLEU by a few tenths | Treat the reference comparison as approximate (±1 BLEU); within-project comparisons use one pipeline |

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| WMT14 EN→DE from the start | Exact paper task but about 1–3 GPU-days per run, so no seeds or ablations within budget |
| A WMT14 subset (for example 1M pairs) as primary | No standard reference numbers for arbitrary subsets; weak external validity |
| Multi30k as primary | Too easy and short (image captions, ~12 words); weak test of long-range dependencies; less credible to reviewers |
| WMT16 EN-RO (~600k pairs) | Good middle ground, but ~4× the cost of IWSLT and fewer from-scratch reference numbers; kept as a possible second benchmark |
| Language modelling (WikiText-2/103, Tiny Shakespeare) | Different task: no encoder, cross-attention, beam search or BLEU. Kept as an optional side track. |

## Compliance
- Every number in `docs/results.md` carries its dataset, split, direction and BLEU protocol/signature.
- The data preparation script refuses to run when checksums do not match.
- Review checklist: "Any comparison with a published number states the protocol on both sides."
