# ADR-017: Loss, normalization and optimizer-update semantics

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** Lead engineer
- **Related:** [math.md §7–8](../math.md), [ADR-013](ADR-013-run-artifacts-and-resumption.md), [ADR-014](ADR-014-numerical-precision.md), spec §6.1

## Context
The paper gives ε_ls = 0.1 and the Noam formula, but not the loss normalization, how gradient accumulation interacts with it, or what a "step" is under mixed precision, where `GradScaler` may skip updates. These details decide whether the loss curve and the learning-rate schedule mean what we think they mean.

## Decision
1. **Loss.** Label-smoothed cross-entropy with the ε/K convention over the full vocabulary (math.md §8.1), computed from **float32** `log_softmax` (ADR-014): `(1−ε)·NLL + ε·mean_k(−log p_k)` per target token. Positions where the target is PAD contribute nothing. It must equal `F.cross_entropy(label_smoothing=ε, ignore_index=PAD, reduction="sum")` (tested).
2. **Normalization: per target token of the whole update.** Each micro-batch contributes `loss_sum / tokens_in_update`, where `tokens_in_update` is summed over **all** micro-batches of the optimizer update (they are gathered before the backward passes). Gradient accumulation is therefore mathematically identical to one large batch (tested), whatever the batch sizes.
3. **Reported metrics:** smoothed loss per token, **unsmoothed NLL per token, perplexity = exp(NLL)** (math.md §8.4), and teacher-forced token accuracy. BLEU arrives in Week 6.
4. **"Step" means a successful optimizer update.** The schedule receives `update + 1` before each `optimizer.step`. When `GradScaler` skips an update because of inf/NaN gradients (detected by the scale shrinking), the update counter, the schedule and the token counter do **not** advance. The skip is counted and logged.
5. **Gradient norm** is computed after `unscale_` on every update and logged. Clipping is off unless `clip_norm > 0` (spec §6.1).
6. **Non-finite guard.** Without a scaler (fp32/bf16), a non-finite loss or gradient norm saves the offending batch to the run directory and raises `FloatingPointError`.

## Consequences
**Positive:** loss and learning rate are comparable across batch sizes, accumulation settings and AMP. Skipped fp16 steps don't distort the warmup. Perplexity is honest. **Negative:** the micro-batches of an update are held on the CPU before their backward passes (small). **Risk:** a long run of skipped steps → the skip count is logged, and a warning is raised if more than 5% of steps are skipped.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Mean per micro-batch, then average over micro-batches | Weights short and long batches equally; accumulation ≠ large batch |
| Per-sentence normalization | Long sentences under-weighted; not the reference practice |
| Count skipped fp16 steps as steps | Shifts the warmup by the number of overflows |
| ε/(K−1) or ε/(K−2) smoothing | Equivalent for BLEU but loses PyTorch as a parity reference (math.md §8.1) |

## Compliance
`tests/unit/test_losses.py` (parity with `F.cross_entropy`, loss floor, padding), `tests/unit/test_schedule.py`, `tests/integration/test_trainer.py` (accumulation equivalence, exact resume).
