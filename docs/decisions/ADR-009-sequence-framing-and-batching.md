# ADR-009: Sequence framing, teacher forcing and token-budget batching

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [ADR-005](ADR-005-sentencepiece-tokenization.md), [ADR-008](ADR-008-mask-convention.md), [paper_notes.md §4.1, §6.1](../paper_notes.md)

## Context
The paper says the decoder's "output embeddings are offset by one position" (§3.1) and that batches contain "approximately 25000 source tokens and 25000 target tokens", grouped "by approximate sequence length" (§5.1). It does not specify BOS/EOS placement, how the shift is implemented, or the batching algorithm (paper_notes U13). Off-by-one shift errors are the most common silent failure in seq2seq code: the loss drops to about 0 while generation is garbage.

## Decision
**Framing** (special IDs from ADR-005: PAD=0, UNK=1, BOS=2, EOS=3):

| Side | Stored sequence | Length |
|---|---|---|
| Source | `x₁ … xₙ EOS` | S = n + 1 |
| Target | `BOS y₁ … yₘ EOS` | m + 2 |

The source has no BOS, because the encoder is not autoregressive and EOS marks the end. This follows fairseq. Content IDs may never be PAD/BOS/EOS (UNK is allowed), which is validated at framing time.

**Teacher forcing**, applied **per example before padding**:

```
tgt_in  = BOS y₁ … yₘ      (target[:-1])  length T = m + 1
tgt_out = y₁ … yₘ EOS      (target[1:])   length T = m + 1
```

Splitting per example (rather than slicing the padded batch) guarantees **`tgt_in` never contains EOS** and that `tgt_in` and `tgt_out` have identical padding patterns. The invariant `tgt_out[:, t] == tgt_in[:, t+1]` holds at every non-pad position.

**Lengths** reported by datasets are **tensor lengths**: `S` for the source and `T = m + 1` for the decoder. Filtering (`min_len ≤ n, m`; `S, T ≤ max_len`; optional length ratio) uses these.

**Token-budget batching** (`TokenBudgetBatchSampler`):
1. Cost of a batch = `num_sentences × max(max S, max T)`, i.e. the **padded** tokens of the larger side. Budget: `max_tokens` (optionally also `max_sentences`).
2. Training: per-epoch RNG seeded from `[seed, epoch]` → random permutation → split into pools of `pool_size` examples → sort each pool by length → greedy packing per pool → shuffle the batch order.
3. Evaluation: global sort by length, no shuffle, deterministic.
4. An example longer than `max_tokens` is a configuration error (raised at construction), never silently dropped.
5. Every batch carries the **dataset indices**, so outputs can be restored to corpus order (ADR-006).

**Storage:** pre-tokenized sequences are kept as flat `int32` arrays plus offsets. This is memory-light and pickles quickly for Windows `spawn` workers. Tensors are `int64` (`torch.long`) only after collation.

## Consequences
**Positive:** the shift is correct by construction and tested; padding waste is low; batches are reproducible (seed, epoch); workers are cheap to start. **Negative:** `max_tokens` bounds *padded max-side* tokens, not the paper's "25k source + 25k target real tokens". For roughly length-balanced pairs these are close, and padding efficiency is logged so the effective token count is known. **Risk:** greedy packing per pool leaves a few small batches at pool boundaries (a slight gradient-noise variation) → acceptable; measured by the padding-efficiency statistic.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| BOS on the source too | No benefit for a non-autoregressive encoder; one wasted position |
| Shift the padded batch (`tgt[:, :-1]`) | Leaves EOS inside `tgt_in` for shorter sequences; padding patterns differ between in and out |
| Fixed sentence-count batches | Variable token counts per step and heavy padding; not what the paper does |
| Global sort + random tie-breaks (fairseq style) | Nearly identical batch composition every epoch; pools give more variety for the same padding cost |
| Drop over-long examples inside the sampler | Silent data loss; filtering belongs to the dataset, with statistics |

## Compliance
`tests/unit/test_example.py`, `test_collate.py` (shift invariants, no EOS in `tgt_in`), `test_sampler.py` (each index exactly once, budget respected, determinism, a hypothesis property test).
