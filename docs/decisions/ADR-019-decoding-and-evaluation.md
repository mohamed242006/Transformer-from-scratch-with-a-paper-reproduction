# ADR-019: Decoding, checkpoint averaging and the evaluation pipeline

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** Lead engineer
- **Related:** [ADR-006](ADR-006-evaluation-protocol.md), [ADR-013](ADR-013-run-artifacts-and-resumption.md), [math.md Appendix A](../math.md), paper §6.1

## Context
The paper decodes with beam size 4, GNMT length penalty α = 0.6 and a maximum output length of input + 50 ("terminate early when possible"), and averages the last 5 (base) or 20 (big) checkpoints. It does not define the beam bookkeeping, the stopping rule, the length counted by the penalty, or how outputs are scored. A first implementation that stopped a sentence after *k* finished hypotheses **truncated 6.3% of copy-task outputs**, because short hypotheses ending in EOS filled the finished list before the correct, longer hypothesis completed. The stopping rule matters.

## Decision
1. **Decoding** (`aiayn.inference.decoding`):
   - Encoder once per batch; the decoder is recomputed over the prefix at each step (**no KV cache** yet: correct, and cheap at sentence length; planned extension).
   - PAD and BOS are never generated. Log-probabilities are computed in ≥ float32. fp16 autocast is used on CUDA ≥ 7.0.
   - Length limit per sentence: `max_len_a · |x| + max_len_b` generated tokens (EOS included), with a = 1 and b = 50 (paper), capped by `max_positions − 1`. EOS is forced at the limit.
   - Ranking score: `log P(Y|X) / lp(|Y|)` with `lp(n) = ((5 + n) / 6)^α`, where **|Y| counts generated tokens including EOS**.
2. **Beam search:** from the *k* live beams take the top 2k candidates; EOS candidates within the top *k* are finalized; the best *k* non-EOS candidates continue. **Stopping rule (exact):** a sentence stops when the best finished normalized score ≥ `best_live_score / lp(limit)` (an upper bound on any score still reachable, since log-probabilities ≤ 0 and lp grows with length), or when no beam is alive, or at the limit. With k = 1 and α = 0 this equals greedy decoding. With k ≥ the number of hypotheses it equals exhaustive search. Both properties are tested.
3. **Checkpoint averaging:** arithmetic mean of floating-point tensors (accumulated in float64, cast back); integer tensors must be identical. Sources: the run's `checkpoints/avg/step_*.pt` (newest N) or explicit files. Average checkpoints from the **converged plateau only**: averaging across an under-trained snapshot hurts (seen on the copy task).
4. **Translation API:** `Translator.from_run(run_dir, checkpoint=best|last|averaged|path)` rebuilds the model from `config.yaml` and the run's own tokenizer, without importing training code. Batches are length-sorted under `beam × sentences × longest ≤ max_tokens`, and outputs are returned in input order.
5. **Evaluation pipeline** (`evaluate_run`, `scripts/evaluate.py`): decode the **raw, unfiltered** split; references are the raw target lines in corpus order. Report cased sacreBLEU (primary, with signature), lowercased BLEU, chrF, exact match, BLEU by source length (words), decode time. Results go to `runs/<id>/eval/<split>.<checkpoint>.<greedy|beamK>.{hyp,json}`.
6. **Layering:** `inference` may import `evaluation` (a leaf on `utils`, so no cycles). Only `evaluation` may import `sacrebleu` (policy test).
7. Deferred: paired bootstrap significance (Week 8, when systems are compared), KV cache (extension), sampling-based decoding (out of scope).

## Consequences
**Positive:** paper-faithful decoding, with correctness checked against brute force; one command evaluates any run with signatures; averaging and the checkpoint choice are explicit. **Negative:** without a KV cache, decoding cost per step grows with the prefix (O(T²) attention recomputation); measured at ~30 s for 1,000 copy sentences on a laptop CPU, far less on a T4. **Risk:** the exact bound can run long when α is large → still capped by the length limit.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Stop after k finished hypotheses (fairseq "stop early") | Truncated outputs on the copy task (93.7% vs 99.4% exact match) |
| Normalize by length^α (simple average) | Not the paper's (GNMT) penalty |
| Score references after tokenization/filtering | Not comparable with sacreBLEU numbers elsewhere (ADR-006) |
| Use Hugging Face `generate` | Not hand-written; different conventions |

## Compliance
`tests/unit/test_decoding.py` (unbatched reference, beam = 1 vs greedy, exhaustive search for α ∈ {0, 0.6, 1}, batch independence, limits), `test_averaging.py`, `test_metrics.py` (vs sacreBLEU), `tests/integration/test_evaluation_pipeline.py`.
