# Architecture Decision Records (ADRs)

An ADR records **one** significant decision: the forces that shaped it, what was decided, what it costs, and what was rejected. ADRs are **immutable once accepted**. To change a decision, write a new ADR that *supersedes* the old one, and update the old one's status line only.

## Index

| ADR | Title | Status | Date |
|---|---|---|---|
| [ADR-001](ADR-001-manual-implementation.md) | Implement the Transformer manually instead of using `nn.Transformer` | Accepted | 2026-11-15 |
| [ADR-002](ADR-002-iwslt14-before-wmt14.md) | Use IWSLT14 DE→EN as the primary benchmark before WMT14 EN→DE | Accepted | 2026-11-15 |
| [ADR-003](ADR-003-post-ln-default.md) | Post-LN is the default normalization placement | Accepted | 2026-11-15 |
| [ADR-004](ADR-004-parity-testing.md) | Numerical parity testing against PyTorch reference implementations is mandatory | Accepted | 2026-11-15 |
| [ADR-005](ADR-005-sentencepiece-tokenization.md) | SentencePiece is allowed for subword tokenization | Accepted | 2026-11-15 |
| [ADR-006](ADR-006-evaluation-protocol.md) | Evaluation protocol: sacreBLEU primary, reference-protocol BLEU secondary | Accepted | 2026-11-15 |
| [ADR-007](ADR-007-configuration-system.md) | Configuration system: frozen dataclasses + YAML with strict validation | Accepted | 2026-11-15 |
| [ADR-008](ADR-008-mask-convention.md) | Boolean masks, True = may attend, key-side only, rank 4 | Accepted | 2026-11-15 |
| [ADR-009](ADR-009-sequence-framing-and-batching.md) | Sequence framing, teacher forcing and token-budget batching | Accepted | 2026-11-15 |
| [ADR-010](ADR-010-compute-platform.md) | Compute platform: local CPU for development, Kaggle GPUs for training | Accepted | 2026-11-15 |
| [ADR-011](ADR-011-initialization.md) | Weight initialization | Accepted | 2026-11-15 |
| [ADR-012](ADR-012-dropout-locations.md) | Dropout locations and defaults (paper text first, else 0.0) | Accepted | 2026-11-15 |
| [ADR-013](ADR-013-run-artifacts-and-resumption.md) | Run artifacts, checkpointing and resumption on Kaggle | Accepted | 2026-11-15 |
| [ADR-014](ADR-014-numerical-precision.md) | Numerical precision policy (fp16 on Kaggle GPUs, fp32 reductions) | Accepted | 2026-11-15 |
| [ADR-015](ADR-015-model-assembly.md) | Model assembly: boundaries, weight tying, attention-weight retention, initialization | Accepted | 2026-11-15 |
| [ADR-016](ADR-016-kaggle-code-and-data-delivery.md) | Deliver code and data to Kaggle as private Kaggle Datasets with a content manifest | Accepted (amended) | 2026-11-15 |
| [ADR-017](ADR-017-loss-and-update-semantics.md) | Loss, normalization and optimizer-update semantics | Accepted | 2026-11-16 |
| [ADR-018](ADR-018-experiment-composition.md) | Experiment composition, compatibility checks and resume-safe config hashing | Accepted | 2026-11-16 |
| [ADR-019](ADR-019-decoding-and-evaluation.md) | Decoding, checkpoint averaging and the evaluation pipeline | Accepted | 2026-11-16 |
| [ADR-020](ADR-020-kaggle-run-persistence.md) | Preserve Kaggle runs: commit-mode sessions + private runs dataset, auto-resume | Accepted | 2026-11-16 |
| [ADR-021](ADR-021-rnn-baselines-and-fair-comparison.md) | RNN baselines (LSTM, GRU+Bahdanau, BiLSTM+Luong) and the fair-comparison protocol | Accepted | 2026-11-17 |

### Planned ADRs (not yet written)
| Planned | Topic | Target week |
|---|---|---|
| ADR-022 | IWSLT14 source pinning and preprocessing | Week 8 |

## Lifecycle
`Proposed` → `Accepted` → (`Deprecated` | `Superseded by ADR-XXX`)

## When to write an ADR
Write one when a decision (a) is hard to reverse, (b) affects more than one package, (c) deviates from the paper, or (d) a reviewer would reasonably ask "why did you do it this way?"

## Template

```markdown
# ADR-XXX: <Decision as a short imperative sentence>

- **Status:** Proposed | Accepted | Deprecated | Superseded by ADR-YYY
- **Date:** YYYY-MM-DD
- **Deciders:** <roles>
- **Related:** <ADRs, docs, spec sections>

## Context
What problem or force requires a decision? Facts, constraints, risks.

## Decision
What we will do, stated precisely enough to verify.

## Consequences
### Positive
### Negative / costs
### Risks and mitigations

## Alternatives considered
| Alternative | Why rejected |
|---|---|

## Compliance
How the decision is enforced or verified (test, CI check, review checklist).
```
