# Study `week9_iwslt14`

Reference: `transformer`. Protocol: ADR-021 (fairness), ADR-022 (study and statistics).

**Status:** 9 runs completed, 0 failed, 0 planned runs not started, 0 tasks pending — complete.

## 1. Main comparison (protocol defaults)

Test set, averaged checkpoint, beam 4, α = 0.6; seeds [1, 2, 3].

| Model | Params | Seeds | **BLEU** mean ± std | 95% CI (seeds) | min–max | chrF | tok-lc BLEU | Len ratio | Valid NLL | ΔBLEU | 95% CI (Δ) | p | p (Holm) | p (seed perm.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer | 36,665,344 | 3 | **32.95 ± 0.05** | ±0.11 | 32.90–32.99 | 56.58 ± 0.07 | 34.67 ± 0.05 | 0.954 ± 0.003 | 1.600 ± 0.002 | ref | ref | ref | ref | ref |
| bilstm_luong | 36,606,112 | 3 | **31.80 ± 0.19** | ±0.47 | 31.63–32.00 | 55.53 ± 0.11 | 33.57 ± 0.18 | 0.962 ± 0.002 | 1.730 ± 0.002 | -1.15 | [-1.47, -0.87] | ≤0.001 | 0.002 | 0.100 |
| gru_bahdanau | 36,565,136 | 3 | **11.51 ± 14.37** | ±35.69 | 0.06–27.63 | 29.51 ± 21.79 | 12.40 ± 15.12 | 1.931 ± 1.412 | 1.931 ± 0.002 | -21.45 | [-32.64, -5.38] | ≤0.001 | 0.002 | 0.100 |

## 2. Comparison with decoding selected on validation

Each model uses its own valid-selected checkpoint and (beam, α) (section 5); everything else as in section 1.

- transformer: averaged checkpoint, beam 8, α 1 ⚠ beam at grid maximum; alpha at grid maximum
- bilstm_luong: averaged checkpoint, beam 8, α 1 ⚠ beam at grid maximum; alpha at grid maximum
- gru_bahdanau: best checkpoint, beam 8, α 1 ⚠ beam at grid maximum; alpha at grid maximum

| Model | Params | Seeds | **BLEU** mean ± std | 95% CI (seeds) | min–max | chrF | tok-lc BLEU | Len ratio | Valid NLL | ΔBLEU | 95% CI (Δ) | p | p (Holm) | p (seed perm.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer (averaged, beam8-a1) | 36,665,344 | 3 | **33.10 ± 0.07** | ±0.18 | 33.02–33.16 | 56.73 ± 0.09 | 34.84 ± 0.08 | 0.958 ± 0.002 | 1.600 ± 0.002 | ref | ref | ref | ref | ref |
| bilstm_luong (averaged, beam8-a1) | 36,606,112 | 3 | **32.04 ± 0.10** | ±0.24 | 31.96–32.15 | 55.78 ± 0.06 | 33.83 ± 0.09 | 0.966 ± 0.002 | 1.730 ± 0.002 | -1.06 | [-1.33, -0.82] | ≤0.001 | 0.002 | 0.100 |
| gru_bahdanau (best, beam8-a1) | 36,565,136 | 3 | **28.88 ± 0.25** | ±0.63 | 28.60–29.09 | 52.87 ± 0.18 | 30.78 ± 0.14 | 0.973 ± 0.008 | 1.931 ± 0.002 | -4.22 | [-4.57, -3.86] | ≤0.001 | 0.002 | 0.100 |

## 3. Comparison with training configuration selected on validation

Per model, the sensitivity variant with the highest mean validation BLEU (NLL breaks ties) (seeds []), run on all main seeds:

- transformer: `default`
- bilstm_luong: `default`
- gru_bahdanau: `default`

| Model | Params | Seeds | **BLEU** mean ± std | 95% CI (seeds) | min–max | chrF | tok-lc BLEU | Len ratio | Valid NLL | ΔBLEU | 95% CI (Δ) | p | p (Holm) | p (seed perm.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer (default) | 36,665,344 | 3 | **32.95 ± 0.05** | ±0.11 | 32.90–32.99 | 56.58 ± 0.07 | 34.67 ± 0.05 | 0.954 ± 0.003 | 1.600 ± 0.002 | ref | ref | ref | ref | ref |
| bilstm_luong (default) | 36,606,112 | 3 | **31.80 ± 0.19** | ±0.47 | 31.63–32.00 | 55.53 ± 0.11 | 33.57 ± 0.18 | 0.962 ± 0.002 | 1.730 ± 0.002 | -1.15 | [-1.44, -0.84] | ≤0.001 | 0.002 | 0.100 |
| gru_bahdanau (default) | 36,565,136 | 3 | **11.51 ± 14.37** | ±35.69 | 0.06–27.63 | 29.51 ± 21.79 | 12.40 ± 15.12 | 1.931 ± 1.412 | 1.931 ± 0.002 | -21.45 | [-32.71, -5.38] | ≤0.001 | 0.002 | 0.100 |

## 4. Training sensitivity

Seeds []; test BLEU at protocol decoding. Same grid for every model (DESIGN.md §7.3). `= default` marks a variant equal to the model's default config.

## 5. Decoding sensitivity

Mean valid BLEU over seeds [1, 2, 3], on each model's selected checkpoint (best vs averaged, chosen on validation first). The valid-best setting is used in section 2; test data never drives a choice.

**transformer** (checkpoint: averaged; valid BLEU averaged 34.61 vs best 34.18; selected: beam 8, α 1 ⚠ beam at grid maximum; alpha at grid maximum)

| Beam | α=0.6 | α=1 |
|---|---|---|
| 4 | 34.61 | 34.70 |
| 8 | 34.65 | 34.79 |

**bilstm_luong** (checkpoint: averaged; valid BLEU averaged 33.38 vs best 32.61; selected: beam 8, α 1 ⚠ beam at grid maximum; alpha at grid maximum)

| Beam | α=0.6 | α=1 |
|---|---|---|
| 4 | 33.38 | 33.51 |
| 8 | 33.52 | 33.65 |

**gru_bahdanau** (checkpoint: best; valid BLEU averaged 11.73 vs best 29.94; selected: beam 8, α 1 ⚠ beam at grid maximum; alpha at grid maximum)

| Beam | α=0.6 | α=1 |
|---|---|---|
| 4 | 29.94 | 29.98 |
| 8 | 30.14 | 30.21 |

## 6. Checkpoint averaging and beam search

Paired on the same runs (all main seeds); Δ = first − second; Holm over the whole table.

| Model | Comparison | Δ BLEU | 95% CI (Δ) | p | p (Holm) |
|---|---|---|---|---|---|
| transformer | averaged − best (beam 4) | +0.45 | [+0.26, +0.63] | ≤0.001 | 0.006 |
| transformer | beam 4 − greedy (best) | +0.85 | [+0.62, +1.08] | ≤0.001 | 0.006 |
| bilstm_luong | averaged − best (beam 4) | +0.73 | [+0.41, +1.04] | ≤0.001 | 0.006 |
| bilstm_luong | beam 4 − greedy (best) | +1.44 | [+1.17, +1.72] | ≤0.001 | 0.006 |
| gru_bahdanau | averaged − best (beam 4) | -17.06 | [-28.30, -1.10] | ≤0.001 | 0.006 |
| gru_bahdanau | beam 4 − greedy (best) | +1.61 | [+1.28, +1.91] | ≤0.001 | 0.006 |

## 7. Ablations

Each variant (seeds []) against its model's default (seeds [1, 2, 3]); test, averaged checkpoint, protocol decoding. Holm over all ablations of a model.

## 8. Statistical notes and checks

- Validation NLL vs test BLEU over 9 runs: Spearman ρ = -0.817 (selection on validation NLL is meaningful only if this is strongly negative).
- Variance sources (main runs): seed-to-seed BLEU std 4.87; test-sampling std of a single run 0.21. Single-seed confidence intervals capture only the second.
- PASS: one tokenizer fingerprint across all runs.
- PASS: one data configuration across all runs (micro-batch size aside).
- PASS: one effective update size (max_tokens x update_freq): 8,192 tokens.
- Declared differences: RNN experiments use `train.clip_norm=1.0` (ADR-021); memory-bound models may accumulate gradients at an identical effective update size (ADR-023 §8); variants differ only by their listed overrides; seeds by `train.seed`.
- Tests: two-level paired bootstrap (1000 samples; sentences × seeds), p floor ≤0.001; Holm-Bonferroni within each table; exact seed permutation test as a training-variance-only check (smallest attainable p: 2/252 for 5 vs 5 seeds; 1/56 for 3 vs 5 seeds).

## 9. Training health and outlier seeds

Means over seeds. Flags: best validation before 60% of training (over-fitting or instability); final train NLL more than 0.3 above the model default's (under-fitting or optimization failure).

| Model | Configuration | Seeds | Best at | Final train NLL | Best valid NLL | Max grad norm | Flags |
|---|---|---|---|---|---|---|---|
| transformer | default | 3 | 93% | 1.100 ± 0.002 | 1.600 ± 0.002 | 1.9 |  |
| bilstm_luong | default | 3 | 47% | 1.187 ± 0.003 | 1.730 ± 0.002 | 1.3 | early best |
| gru_bahdanau | default | 3 | 44% | 11.640 ± 8.206 | 1.931 ± 0.002 | 9.8 | early best |

**Outlier seeds** (test BLEU more than 3 below their configuration's median; possible silent collapse):

- `iwslt14_gru_bahdanau_s1`: 0.06 vs median 6.84
