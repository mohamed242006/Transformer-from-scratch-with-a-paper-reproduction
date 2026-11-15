# ADR-012: Dropout locations and defaults

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [paper_notes.md §4.5, §6.4, U2/U3](../paper_notes.md), spec §6

## Context
The paper specifies two dropout sites for the MT models (§5.4): **residual dropout** (on each sub-layer output, before the residual add and normalization) and dropout on **embedding + positional-encoding sums**, with P_drop = 0.1 (base). Attention dropout is mentioned only in passing for parsing (§6.3). Dropout inside the FFN is never mentioned. Week 1 provisionally set attention dropout to 0.1 "to match reference implementations" and FFN dropout to 0.0 "paper-literal". That mixed two principles, which the Week 1 self-review flagged.

## Decision
**Principle: follow the paper text. Where the paper is silent, default to 0.0 and make the value configurable and ablatable.**

| Site | Config key (Week 4) | Default | Source |
|---|---|---|---|
| Embedding + PE sum | `dropout` | P_drop | Paper §5.4 |
| Residual (sub-layer output, before add) | `dropout` | P_drop | Paper §5.4 |
| Attention weights (after softmax) | `attention_dropout` | **0.0** | Paper silent for MT |
| FFN inner activation (after ReLU) | `activation_dropout` | **0.0** | Paper silent |

Modules expose these as separate constructor arguments: `SinusoidalPositionalEncoding(dropout=…)`, `SublayerConnection(dropout=…)`, `MultiHeadAttention(dropout=…)` (attention weights) and `PositionwiseFeedForward(dropout=…)` (inner activation). Attention dropout applies to the weights multiplied with V. The weights *returned* for analysis are the pre-dropout probabilities.

Ablation (Multi30k, ADR-010): attention dropout ∈ {0.0, 0.1}, activation dropout ∈ {0.0, 0.1}.

## Consequences
**Positive:** a single, defensible rule; matches the paper text and common IWSLT recipes (fairseq's IWSLT configuration uses `dropout=0.3` with attention and activation dropout off by default). Reviewers can trace every dropout to a sentence in the paper. **Negative:** supersedes the Week 1 provisional 0.1 attention-dropout default (paper_notes U2 updated). It may under-regularize the big-data setting, but that setting is out of scope (WMT14 removed). **Risk:** overfitting on small data → covered by the mid config's P_drop = 0.3 and the ablation.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Copy tensor2tensor "v2" defaults (attention/ReLU dropout 0.1) | A later codebase revision, not the paper |
| One shared dropout value everywhere (Annotated Transformer style) | Hides which sites matter; not paper-literal |
| No attention dropout support at all | Prevents the ablation and the parsing-style setting |

## Compliance
Unit tests check each site independently: dropout active only in `train()`, applied at the documented position, identity at p = 0.
