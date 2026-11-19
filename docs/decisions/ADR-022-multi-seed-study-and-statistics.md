# ADR-022: Multi-seed study, sensitivity, ablations and statistical protocol (Week 8)

- **Status:** Accepted
- **Date:** 2026-11-17
- **Deciders:** Lead engineer, project owner
- **Related:** [ADR-002](ADR-002-iwslt14-before-wmt14.md), [ADR-006](ADR-006-evaluation-protocol.md), [ADR-020](ADR-020-kaggle-run-persistence.md), [ADR-021](ADR-021-rnn-baselines-and-fair-comparison.md), [DESIGN.md §7.3, §13](../DESIGN.md)

## Context
Week 7 produced a single-seed comparison on Multi30k test 2016:

| Model | BLEU | 95% CI (test sampling) | ΔBLEU | p |
|---|---|---|---|---|
| Transformer | 36.87 | ±1.63 | | |
| BiGRU + Bahdanau | 34.14 | | −2.73 | ≤ 0.001 |
| BiLSTM + Luong | 32.65 | | −4.22 | ≤ 0.001 |
| LSTM without attention | 12.28 | | | |

That evidence has three gaps:
1. **Training variance.** The CIs cover test-set sampling only. Seed-to-seed variance is unmeasured, so a 2.7 BLEU gap could partly be seed luck.
2. **Hyperparameters.** Every model used one learning rate and one dropout. The RNNs produce long outputs (length ratio 1.02–1.09) while the Transformer's are short (0.963), so the shared α = 0.6 suits the models differently.
3. **Mechanism.** It is untested which Transformer components matter at this scale (paper Table 3).

The owner scoped Week 8 as multi-seed experiments, sensitivity studies, ablations, statistical analysis and final experimental reporting. The original roadmap had Week 8 as the IWSLT14 benchmark and Week 9 as ablations. This ADR records the change: Week 8 runs on **Multi30k**, and the IWSLT14 headline (ADR-002) remains planned as ADR-023.

## Decision
1. **A declarative study** (`configs/study/week8.yaml`) defines every run, evaluation and selection rule. `aiayn.study` expands it, executes it within a session budget, and reports on it. Nothing is chosen by hand after seeing results.
2. **Design** (same grids for every model, per DESIGN.md §7.3):

   | Part | Runs | Seeds | Purpose |
   |---|---|---|---|
   | Main | 4 models, protocol defaults | 1–5 | Headline comparison with training variance |
   | Decoding grid | beam {1, 4, 8} × α {0, 0.3, 0.6, 1.0, 1.5}, averaged checkpoint, **valid** | 1–3 | Per-model decoding sensitivity |
   | Decoding-selected | Valid-best (beam, α) per model, on test | 1–5 | Comparison under each model's own best decoding |
   | Training sensitivity | peak LR ×0.5/×2; dropout {0.1, 0.2, 0.3, 0.4} | 1–3 | LR and regularization sensitivity |
   | Tuned | Lowest mean **valid** NLL variant per model, topped up to all main seeds | 1–5 | Comparison under each model's own best training config |
   | Ablations | Transformer variants (below) | 1–3 | Which components matter |

   The ten ablations are:
   - (A) 1 head, 8 heads;
   - (C) 1+1 layers, 6+6 layers;
   - (D) no dropout, no label smoothing;
   - (E) learned positional embeddings;
   - Pre-LN (ADR-003), untied embeddings, no √d embedding scale.

   Variants equal to a model's default config (same config hash) reuse its main runs. The Week 6/7 seed-1 runs are reused by name.
3. **No selection on test.** Decoding is selected by mean valid BLEU and training configs by mean best valid NLL. Test numbers are reported for every variant but never drive a choice. Ties keep the protocol default.
4. **Statistics:**
   - Report mean ± std and a 95% Student-t interval over seeds.
   - The **primary test** is a two-level paired bootstrap:
     - It resamples test sentences, shared by both systems so the test is paired, and each system's seeds with replacement.
     - Corpus BLEU on every resample comes from sacreBLEU's own sufficient statistics, so it equals sacreBLEU exactly (tested).
     - It reports Δ = mean-over-seeds BLEU difference, a 95% percentile CI and a centred two-sided p-value. With B = 1000 samples the p floor is 1/1001 ("≤ 0.001").
   - **Holm-Bonferroni** correction is applied within each table, the family of comparisons.
   - **Supplementary:** an exact seed permutation test. It measures training variance only, and its smallest attainable p depends on the seed counts (5 vs 5: 2/252 ≈ 0.008; 3 vs 5: 1/56 ≈ 0.018 when the result is extreme in one direction, 2/56 ≈ 0.036 only in the symmetric case). *(Corrected 2026-11-18; the original text said 3 vs 5: 0.036.)*
   - **Diagnostics:** the Spearman correlation between valid NLL and test BLEU (it validates NLL-based selection), and the seed std compared with the test-sampling std.
5. **Execution on Kaggle** (`notebooks/kaggle_study.ipynb`, `scripts/run_study.py`):
   - Work proceeds in priority order: main → decoding → decoding-selected → ablations → sensitivity → tuned. A session that runs out of time leaves the most important results complete.
   - Training runs as a `scripts/train.py` subprocess with the remaining time as its budget; evaluation runs in-process.
   - A run that diverges or crashes is marked `FAILED.json` and reported, never retried. For selection it counts as +inf NLL.
   - Every study run is copied into the working directory, so each session's output, and the `aiayn-runs` version published from it, is a complete snapshot.
6. **Storage:**
   - Completed runs are slimmed to `averaged.pt`, plus weights-only `best.pt` for main runs.
   - This takes ~330 MB down to ~30–60 MB per run, which keeps about 110 runs (~4 GB) under Kaggle's 20 GB output limit.
   - Slimmed runs remain fully evaluable (identical BLEU, tested) but are not resumable. Only completed runs are slimmed.

## Consequences
**Positive:**
- Every claim in the final report carries training variance, test-sampling variance and a multiple-comparison correction.
- Model selection is reproducible and test-blind.
- The study survives session limits and failures.

**Negative:**
- About 9 h of T4 time (one or two sessions).
- 3 seeds for sensitivity and ablations give wide intervals; effects under ~1 BLEU will usually not be significant. The report says so rather than over-claiming.
- Optimizer states of completed runs are discarded.

**Not covered:**
- The IWSLT14 benchmark (ADR-023, planned).
- RNN ablations (input feeding, source reversal; ADR-021 §3).
- Figures (Week 10).

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Single-level bootstrap per seed | Ignores training variance, the gap Week 7 left open |
| Seed-level t-test only | 3–5 points; ignores test-set sampling; distributional assumptions |
| Full LR × dropout grid | 9 configs × 4 models × 3 seeds ≈ 108 runs for sensitivity alone; one-at-a-time answers the question at half the cost |
| Selecting on test BLEU | Optimistic bias; invalidates the comparison |
| Keeping full checkpoints | Exceeds the 20 GB output limit at ~110 runs |
| Hand-written run lists in the notebook | Error-prone at 110 runs; no automatic follow-up phases |

## Compliance
- `tests/unit/test_statistics.py` covers:
  - bootstrap BLEU equals sacreBLEU;
  - calibration on identical and very different systems;
  - the effect of seed variance;
  - exact permutation values, Holm, Spearman with ties, t intervals.
- `tests/unit/test_study_plan.py` covers:
  - plan shape, aliases and grid tags;
  - config validation, including that test may not be used for selection;
  - selection rules, failure handling and task ordering.
- `tests/integration/test_study_runner.py` covers:
  - the full study with fake train/eval;
  - budget stop and continuation;
  - resumable exits, failures, and adoption of runs from inputs;
  - report sections;
  - the train command line;
  - slimming a really trained run leaves BLEU unchanged.

## Outcome and amendments (2026-11-18)
- **Executed:** 118/118 runs completed, none failed. Results are in [results.md](../results.md) §3–§5.
- **Correction:** the minimum attainable permutation p for 3 vs 5 seeds is 1/56 ≈ 0.018. The original text said 0.036, which holds only for symmetric cases.
- **Lessons adopted for later studies** (ADR-023, implemented in `aiayn.study`):
  1. Training configs are selected by **validation BLEU**, with NLL only as a tie-breaker. Here, validation NLL picked dropout 0.2 over LR ×2 by 0.0002 nats, a coin flip.
  2. The report **flags selections at a grid edge**: all four learning-rate selections were at the edge.
  3. The checkpoint (best or averaged) is **selected per model on validation**, because averaging over a wide window hurt the RNNs by 0.4–1.7 BLEU.
  4. **Outlier seeds** (silent collapse, as with BiLSTM + Luong seed 3) are flagged.
  5. A training-health table (best-update position, final train vs best validation NLL) makes optimization failures such as `layers6` visible.
- `configs/study/week8.yaml` pins the Week 8 rules (`training_selection: valid_nll`, `select_checkpoint: false`) so its report remains reproducible.
