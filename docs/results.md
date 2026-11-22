# Results

Final results of the v1.0 reproduction. Two benchmarks:
- **IWSLT'14 DE→EN** (Week 9; [ADR-023](decisions/ADR-023-iwslt14-benchmark.md)) is the **headline**: 160k sentence pairs and ≈ 36.6M-parameter models.
- **Multi30k EN→DE** (Weeks 7–8; [ADR-021](decisions/ADR-021-rnn-baselines-and-fair-comparison.md), [ADR-022](decisions/ADR-022-multi-seed-study-and-statistics.md)) is the small-scale study with ablations: 29k sentence pairs and 7.6M-parameter models.

**Provenance:**
- **IWSLT'14:** evidence is archived in [`experiments/2026-11-22_iwslt14-week9-benchmark/`](../experiments/2026-11-22_iwslt14-week9-benchmark/NOTES.md): the study report, the configs, training logs and evaluations of all 24 runs, and test hypotheses.
- **Multi30k:** the numbers come from preserved runs in the private `aiayn-runs` dataset (ADR-020). `scripts/compare_runs.py` produces Week 7's `COMPARISON.md`, and `scripts/analyze_study.py --study configs/study/week8.yaml` produces Week 8's `STUDY.md` and `STUDY.json`.

**Metric:** sacreBLEU, cased and detokenized, unless stated otherwise. Every selection (learning rate, checkpoint, decoding) was made on validation data; the test set never drove a choice.

Contents:
- **Part I, IWSLT'14:** [1. Headline](#1-headline-iwslt14-deen) · [2. IWSLT'14 benchmark](#2-iwslt14-benchmark-adr-023)
- **Part II, Multi30k:** [3. Headline](#3-multi30k-headline) · [4. Week 7: baselines](#4-week-7-rnn-baselines-adr-021) · [5. Week 8: study](#5-week-8-multi-seed-study-adr-022) · [6. Ablations](#6-transformer-ablations) · [7. Statistics](#7-statistical-protocol-and-what-is-significant) · [8. Limitations](#8-limitations-and-design-lessons-multi30k) · [9. Conclusions](#9-conclusions-multi30k)
- [10. Project conclusions](#10-project-conclusions) · [11. Future work](#11-future-work)

---

# Part I: IWSLT'14 DE→EN

## 1. Headline: IWSLT'14 DE→EN

Test set (6,750 sentences, identical to fairseq's); 3 seeds per model; mean ± std.

| Model (parameters matched within 0.3%) | Protocol defaults: test BLEU | Validation-selected: test BLEU | chrF | tok-lc BLEU | Valid NLL |
|---|---|---|---|---|---|
| Transformer, **Pre-LN** 6+6, d = 512 (36.67M) | **32.95 ± 0.05** | **33.10 ± 0.07** | 56.73 | 34.84 | 1.600 |
| BiLSTM + Luong attention (36.61M) | 31.80 ± 0.19 | 32.04 ± 0.10 | 55.78 | 33.83 | 1.730 |
| BiGRU + Bahdanau attention (36.57M) | 11.51 ± 14.37 † | 28.88 ± 0.25 | 52.87 | 30.78 | 1.931 |

- **Column definitions:**
  - *Protocol defaults:* averaged checkpoint (last 5 snapshots), beam 4, α = 0.6.
  - *Validation-selected:* each model's checkpoint and (beam, α) chosen on validation (§2.3).
  - chrF and tok-lc BLEU are for the validation-selected setting. Tok-lc BLEU is the fairseq-style tokenized lowercased BLEU (Protocol B, ADR-023 §4).
- **The Transformer is the best model under both protocols.** Its lead over the best RNN, BiLSTM + Luong, is **1.06 BLEU** (95% CI [−1.33, −0.82]) validation-selected and 1.15 at the protocol defaults. Every gap is significant: two-level paired bootstrap, Holm p = 0.002.
- **The lead is modest.** Each model got its own learning rate, selected at the full budget (§2.1). The ≈ 1 BLEU margin agrees with Multi30k's equal-learning-rate result (1.6–1.8, §5.3), not with its shared-default result (2.7–4.3).
- **Declared deviation:** the headline Transformer uses Pre-LN. The paper's Post-LN 6+6 did not train in this recipe (valid BLEU 5.79 at 30k updates; §2.8). This is bounded, not explained (ADR-023 Amendment 4).
- **† GRU + Bahdanau** diverged late in training in 2 of 3 seeds, so its averaged checkpoint is unusable. The validation-selected best checkpoint gives 28.88 (§2.5).
- **Literature scale:** the Transformer's tok-lc BLEU is 34.67 at the protocol defaults and 34.84 validation-selected. fairseq-reported 6+6 IWSLT'14 Transformers are ≈ 34–35. Those systems train on tokenized lowercased text and ours trains on raw cased text, so this places the result on the literature's scale; it is not an exact match (ADR-023 §4).

## 2. IWSLT'14 benchmark (ADR-023)

### 2.1 Setup
- **Data:** fairseq `prepare-iwslt14.sh` split membership on raw cased text: train 160,215, valid 7,282, test 6,750 pairs. Test is identical to fairseq's; train and valid differ by 25 pairs (ADR-023 §2). Joint SentencePiece BPE, 10k.
- **Models:**

  | Model | Configuration | Parameters |
  |---|---|---|
  | Transformer | 6+6 layers, d = 512, 4 heads, d_ff = 1,024, dropout 0.3, Pre-LN, three-way tied embeddings | 36,665,344 |
  | BiLSTM + Luong | 2 × 720 | 36,606,112 |
  | BiGRU + Bahdanau | 1 × 1,104 | 36,565,136 |

- **Training, identical for all models:**
  - 30,000 updates of 8,192 target tokens;
  - inverse-square-root schedule with 4,000 warmup;
  - label smoothing 0.1; Adam (0.9, 0.98, 1e-9); fp16;
  - snapshots every 1,000 updates, the last 5 averaged.
- **Declared differences (ADR-021, ADR-023 Amendment 1):**
  - The RNNs clip gradients at 1.0.
  - The GRU accumulates 2 × 4,096 tokens per update, for memory; this gives the same effective update.
- **Peak learning rate per model, selected at the full budget on seed-1 validation BLEU (ADR-023 Amendment 3):**
  - Transformer 1e-3, chosen over 7e-4 and 1.4e-3;
  - BiLSTM 5.6e-3, chosen over 2.8e-3 and 1.12e-2;
  - GRU 2.8e-3, selected in the 6k-update pilot only.
- **Selection on validation (ADR-023 §7):** best vs averaged checkpoint per model, then decoding over beam {4, 8} × α {0.6, 1.0}.
- **Hardware:** one RTX 5070 Ti (16 GB). All 24 IWSLT runs (pilot, selection, benchmark) took **8.9 GPU-hours** in total.

### 2.2 Main comparison: protocol defaults

Averaged checkpoint, beam 4, α = 0.6; seeds 1–3.

| Model | BLEU | 95% CI (seeds) | min–max | chrF | tok-lc BLEU | Len ratio | ΔBLEU [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Transformer (Pre-LN) | **32.95 ± 0.05** | ±0.11 | 32.90–32.99 | 56.58 | 34.67 | 0.954 | ref | |
| BiLSTM + Luong | 31.80 ± 0.19 | ±0.47 | 31.63–32.00 | 55.53 | 33.57 | 0.962 | −1.15 [−1.47, −0.87] | 0.002 |
| BiGRU + Bahdanau | 11.51 ± 14.37 | ±35.69 | 0.06–27.63 | 29.51 | 12.40 | 1.931 | −21.45 [−32.64, −5.38] | 0.002 |

### 2.3 Validation-selected checkpoint and decoding

| Model | Checkpoint (valid BLEU: averaged vs best) | Decoding | Test BLEU | chrF | Len ratio | ΔBLEU [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|
| Transformer (Pre-LN) | averaged (34.61 vs 34.18) | beam 8, α 1.0 | **33.10 ± 0.07** | 56.73 | 0.958 | ref | |
| BiLSTM + Luong | averaged (33.38 vs 32.61) | beam 8, α 1.0 | 32.04 ± 0.10 | 55.78 | 0.966 | −1.06 [−1.33, −0.82] | 0.002 |
| BiGRU + Bahdanau | **best** (11.73 vs 29.94) | beam 8, α 1.0 | 28.88 ± 0.25 | 52.87 | 0.973 | −4.22 [−4.57, −3.86] | 0.002 |

Mean validation BLEU over the decoding grid, on each model's selected checkpoint:

| Model | beam 4, α 0.6 | beam 4, α 1.0 | beam 8, α 0.6 | beam 8, α 1.0 |
|---|---|---|---|---|
| Transformer | 34.61 | 34.70 | 34.65 | **34.79** |
| BiLSTM + Luong | 33.38 | 33.51 | 33.52 | **33.65** |
| BiGRU + Bahdanau | 29.94 | 29.98 | 30.14 | **30.21** |

- **Every model selected the corner of the grid** (largest beam and α), so the optimum was not bracketed.
- **The effect is small, though:** +0.18 to +0.27 on validation, and +0.15 (Transformer) or +0.24 (BiLSTM) on test.
- **The ranking is the same under every grid setting.**

### 2.4 Per-seed test BLEU

| Model | Seed | Averaged, beam 4 | Best, beam 4 | Best, greedy | Validation-selected |
|---|---|---|---|---|---|
| Transformer | 1 / 2 / 3 | 32.99 / 32.90 / 32.96 | 32.55 / 32.42 / 32.56 | 31.71 / 31.51 / 31.77 | 33.12 / 33.02 / 33.16 |
| BiLSTM + Luong | 1 / 2 / 3 | 31.63 / 31.76 / 32.00 | 31.12 / 30.85 / 31.22 | 29.55 / 29.58 / 29.75 | 31.96 / 32.01 / 32.15 |
| BiGRU + Bahdanau | 1 / 2 / 3 | **0.06** / 27.63 / **6.84** | 28.27 / 28.76 / 28.67 | 26.90 / 27.05 / 26.93 | 28.60 / 29.09 / 28.96 |

### 2.5 GRU + Bahdanau: late-training divergence

- **All seeds reached their best validation NLL at updates 13–14k** (1.928–1.932).
- **After that, two seeds diverged:**
  - seed 1: training NLL rose above 3 at about update 22,800 and ended at 15.8; validation NLL was 11.35 at 25k;
  - seed 3: training NLL rose above 3 at about update 27,600 and ended at 17.0;
  - seed 2 drifted without diverging: validation NLL went from 1.94 to 2.03.
- **The averaged checkpoint (updates 26–30k) therefore inherits the divergence.**
- **Not caused by fp16:** only 23–36 updates per run were skipped by the loss scaler.
- **How the study caught it:**
  - the training-health table flags *early best*;
  - the outlier-seed check flags seed 1;
  - choosing the checkpoint on validation (best: 29.94 vs averaged: 11.73) recovers a sound model.
- **Cause not investigated** (future work). The pilot-selected learning rate, 2.8e-3, is the first suspect: unlike the other models', it was never checked at full budget.

### 2.6 BLEU by source length

Test BLEU per source-length bucket (words), mean over seeds. Validation-selected checkpoint, beam 4, α 0.6. The buckets come from the evaluation JSONs.

| Source words (sentences) | Transformer | BiLSTM + Luong | BiGRU + Bahdanau (best) | Transformer − BiLSTM |
|---|---|---|---|---|
| 0–9 (2,137) | 35.89 | 34.58 | 31.82 | +1.31 |
| 10–19 (2,662) | 34.38 | 33.21 | 29.90 | +1.17 |
| 20–29 (1,177) | 32.83 | 31.46 | 28.11 | +1.37 |
| 30–39 (455) | 30.67 | 29.78 | 26.64 | +0.89 |
| 40+ (319) | 30.01 | 28.95 | 25.39 | +1.06 |

- **Every model degrades with length** by 5–6 BLEU from the shortest to the longest bucket.
- **The Transformer's lead stays roughly constant;** there is no sign that attention RNNs collapse on long sentences at this scale.
- **These are descriptive only:** the per-bucket differences were not tested, and the long buckets are small.

### 2.7 Checkpoint averaging, beam search, cost

Paired on the same runs (seeds 1–3); Holm over the table.

| Model | Averaged − best (beam 4) | Beam 4 − greedy (best) |
|---|---|---|
| Transformer | +0.45 [+0.26, +0.63], p 0.006 | +0.85 [+0.62, +1.08], p 0.006 |
| BiLSTM + Luong | +0.73 [+0.41, +1.04], p 0.006 | +1.44 [+1.17, +1.72], p 0.006 |
| BiGRU + Bahdanau | −17.06 [−28.30, −1.10], p 0.006 | +1.61 [+1.28, +1.91], p 0.006 |

Training health and cost (means over seeds; throughput and memory from seed 1):

| Model | Best valid at | Best valid NLL | Final train NLL | Max grad norm | Train time | Target tok/s (train) | Peak memory (MiB) | Test decoding (sent/s, beam 4) |
|---|---|---|---|---|---|---|---|---|
| Transformer | 93% | 1.600 | 1.100 | 1.9 | 0.54 h | 111k | 4,693 | 215 |
| BiLSTM + Luong | 47% (early best) | 1.730 | 1.187 | 1.3 | 0.32 h | 188k | 2,143 | 521 |
| BiGRU + Bahdanau | 44% (early best) | 1.931 | 11.64 ± 8.21 | 9.8 | 0.88 h | 70k | 8,639 | 515 |

- **Averaging helps the Transformer and BiLSTM.** At IWSLT scale the window covers the last 17% of training. Compare Multi30k, where a 60% window hurt the RNNs (§5.5).
- **Beam search helps every model.**
- **The BiLSTM peaks early** but doesn't degrade.
- **Speed:** cuDNN's LSTM trains and decodes faster than the hand-written Transformer. The GRU's Bahdanau loop runs in Python, and it is the slowest and most memory-hungry model to train.

### 2.8 Post-LN vs Pre-LN (bounded result; future work)

| Run (6+6, d = 512) | Updates | Peak LR | Best valid NLL | BLEU, averaged checkpoint, beam 4 |
|---|---|---|---|---|
| Post-LN | 6k (pilot) | 5e-4 / 7e-4 / 1e-3 | 3.51 / 3.54 / 3.66 | 4.40 / 3.71 / 2.85 (test) |
| Pre-LN | 6k (pilot) | 7e-4 | 1.93 | 26.42 (test) |
| Post-LN | 30k | 5e-4 | 3.034 (at 22k) | 5.79 (valid) |
| Pre-LN | 30k | 7e-4 | 1.618 | 34.33 (valid) |

- **Ruled out:**
  - a learning-rate choice between 5e-4 and 1e-3;
  - too few updates: five times the pilot budget brings NLL only from 3.54 to 3.03;
  - fp16 or divergence: 2 skipped updates, maximum gradient norm 2.95.
- **An implementation fault is unlikely:**
  - the same modules train 3+3 Post-LN on Multi30k to 36.71 BLEU, better than Pre-LN there (§6);
  - they are parity-tested against PyTorch's reference Transformer (ADR-004).
- **Remaining candidates** are recipe and initialization differences from fairseq's `transformer_iwslt_de_en`, which trains this architecture successfully:
  - Adam ε;
  - weight decay;
  - Q/K/V Xavier gain;
  - tokens per update;
  - three-way tying with √d embedding scaling.

  None was tested in v1.0 (ADR-023 Amendment 4).

### 2.9 Statistical notes
- **Primary test:** a two-level paired bootstrap (1,000 samples over sentences × seeds), with Holm correction within each table. The CIs and p-values include both test-sampling and seed variance.
- **Seed permutation test:** with 3 vs 3 seeds its smallest attainable p is 0.100, which every comparison reaches. It cannot show significance at this seed count, and is reported for completeness only.
- **Validation NLL vs test BLEU over the 9 runs:** Spearman ρ = −0.817.
- **Fairness checks, all PASS:**
  - one tokenizer fingerprint;
  - one data configuration;
  - one effective update size (8,192 tokens).

### 2.10 Limitations (IWSLT'14)
1. **The headline Transformer is Pre-LN, not the paper's Post-LN.** The Post-LN failure is bounded but not explained.
2. **The GRU's late divergence** makes its protocol-defaults number unusable, and its learning rate was never checked at full budget. Conclusions about the GRU rest on the validation-selected row.
3. **Learning rates were selected on one seed and a 3-point grid.** All Transformer and BiLSTM selections were interior points of their grids.
4. **The decoding grid was not bracketed:** the corner was chosen for every model.
5. **Three seeds:** enough for the bootstrap, not for the permutation test.
6. **One language pair, one direction, one budget** (30k updates, one GPU). WMT'14, the paper's benchmark, was not run.
7. **Membership:** train and valid differ from fairseq's by 25 pairs, and tok-lc BLEU is comparable to the literature, not identical (ADR-023 §4).

---

# Part II: Multi30k EN→DE

**Common setup (Part II):**
- **Data:** Multi30k Task 1, EN→DE, test 2016 (1,000 sentences); raw cased text; joint SentencePiece BPE 8k.
- **Training recipe** (`multi30k_t4`): 30 epochs, token-budget batches of 4,096 tokens per side, inverse-sqrt schedule (peak 7e-4, warmup 400), label smoothing 0.1, Adam (0.9, 0.98, 1e-9).
- **Decoding:** beam 4, α = 0.6, max length |x| + 50.
- **Metric:** sacreBLEU, cased and detokenized (`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`).
- **Hardware:** a single Kaggle T4, fp16.
- **Unless stated otherwise:** averaged checkpoint (mean of the last 5 snapshots).

## 3. Multi30k headline

| Model (≈ 7.6M parameters, matched within 1%) | Protocol defaults, test BLEU (5 seeds) | Valid NLL | LR ×2, test BLEU (3 seeds) |
|---|---|---|---|
| Transformer (Post-LN, 3+3 layers, d = 256) | **36.71 ± 0.20** | 1.918 | **37.95** |
| BiGRU + Bahdanau attention | 33.97 ± 0.15 | 2.210 | 36.32 |
| BiLSTM + Luong attention | 32.46 ± 0.51 | 2.355 | 36.16 |
| LSTM encoder-decoder, no attention | 12.60 ± 0.26 | 3.071 | 17.73 |

- **Under identical training, the Transformer is the best model.**
  - Under the protocol defaults, every gap is significant: two-level paired bootstrap, Holm-adjusted p ≈ 0.003.
  - With a learning rate tuned for all models (×2), the gap to the attention RNNs narrows to about **1.6–1.8 BLEU**. That comes from 3 seeds and wasn't formally tested (§5.3).
- **Attention is the decisive component:** without it, the encoder's fixed-size state costs 19–24 BLEU.

## 4. Week 7: RNN baselines (ADR-021)

The baselines use the same data, tokenizer, batching, schedule, epochs, label smoothing, checkpoint averaging, decoding and scoring as the Transformer. The trainer, beam search and evaluation code are literally shared, through the `EncoderDecoder` interface. The one declared difference is `clip_norm = 1.0` for the RNNs. Parameter counts are within 1% of the Transformer's 7,577,600.

Seed-1 results (`COMPARISON.md`, all fairness checks PASS):

| Model | Params | Valid NLL | BLEU | 95% CI (test sampling) | ΔBLEU | p | chrF | Len ratio | Train time | Target tokens/s |
|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | 7,577,600 | 1.910 | **36.87** | ±1.63 | ref | | 60.74 | 0.963 | 4.1 min | 62,797 |
| BiGRU + Bahdanau | 7,513,536 | 2.212 | 34.14 | ±1.58 | −2.73 | ≤ 0.001 | 59.21 | 1.039 | 5.3 min | 47,942 |
| BiLSTM + Luong | 7,580,496 | 2.343 | 32.65 | ±1.60 | −4.22 | ≤ 0.001 | 56.65 | 1.024 | 2.8 min | 99,054 |
| LSTM, no attention | 7,536,384 | 3.072 | 12.28 | ±0.85 | −24.59 | ≤ 0.001 | 35.42 | 1.094 | 2.4 min | 115,076 |

**Speed:** at this tiny scale the cuDNN LSTMs are faster per token than the hand-written Transformer. The Bahdanau decoder is slowest because it loops over time steps in Python. Week 8 confirmed every Week 7 number to within 0.3 BLEU of its 5-seed mean.

## 5. Week 8: multi-seed study (ADR-022)

The study is `configs/study/week8.yaml`: 118 runs, all completed, none failed. Selections were made on validation data only.

### 5.1 Main comparison: protocol defaults, 5 seeds

| Model | BLEU | min–max | chrF | Len ratio | ΔBLEU [95% CI] | p (Holm) | p (seed permutation) |
|---|---|---|---|---|---|---|---|
| Transformer | **36.71 ± 0.20** | 36.49–36.96 | 60.74 | 0.965 | ref | | |
| BiGRU + Bahdanau | 33.97 ± 0.15 | 33.79–34.14 | 59.20 | 1.041 | −2.74 [−3.69, −1.84] | 0.003 | 0.008 |
| BiLSTM + Luong | 32.46 ± 0.51 | 31.59–32.90 | 56.36 | 1.021 | −4.25 [−5.24, −3.22] | 0.003 | 0.008 |
| LSTM, no attention | 12.60 ± 0.26 | 12.28–12.99 | 35.69 | 1.089 | −24.11 [−25.35, −22.78] | 0.003 | 0.008 |

- **Seed-to-seed spread is small:** standard deviation 0.15–0.51 BLEU, so each gap is many standard deviations wide.
- **chrF and validation NLL give the same ranking.**

### 5.2 Decoding sensitivity: selected on validation, reported on test

Grid: beam {1, 4, 8} × α {0, 0.3, 0.6, 1.0, 1.5}, on the validation set, seeds 1–3.

| Model | Valid-selected | Valid gain | Test BLEU at selected (5 seeds) | Test change vs default | Gap to Transformer (p Holm) |
|---|---|---|---|---|---|
| Transformer | beam 4, α 1.5 (grid edge) | +0.76 | 36.58 ± 0.55 | −0.14 | ref |
| BiGRU + Bahdanau | beam 8, α 0.6 | +0.47 | 34.41 ± 0.22 | +0.44 | −2.17 (0.003) |
| BiLSTM + Luong | beam 4, α 1.0 | +0.03 | 32.07 ± 0.53 | −0.38 | −4.50 (0.003) |
| LSTM, no attention | beam 8, α 0.0 | +0.55 | 12.93 ± 0.25 | +0.33 | −23.64 (0.003) |

- **Decoding is second-order:** test changes stay within ±0.5 BLEU, and validation gains don't reliably carry over to test.
- **The ranking is identical under every decoding setting.**
- **The Transformer's outputs run short:** it prefers a large α (length ratio 0.965 at α 0.6, 1.01–1.03 at α 1.5).
- **The RNNs fall apart at α 1.5** (−2.4 to −3.3 BLEU on validation).

### 5.3 Training sensitivity: one factor at a time, seeds 1–3, mean test BLEU

| Model | LR ×0.5 | default | **LR ×2** | dropout 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|---|
| Transformer (default dropout 0.1) | 33.97 | 36.77 | **37.95** | = default | 36.67 | 35.06 | 30.67 |
| BiLSTM + Luong (default 0.3) | 19.59* | 32.25 | **36.16** | 27.26* | 29.98* | = default | 31.39 |
| BiGRU + Bahdanau (default 0.3) | 28.54 | 34.00 | **36.32** | 34.23 | 34.01 | = default | 33.19 |
| LSTM, no attention (default 0.3) | 9.13 | 12.49 | **17.73** | 13.27 | 13.11 | = default | 12.06 |

\* includes a collapsed seed (§5.4).

- **Learning rate dominates.** For every model the best value is the edge of the grid (×2), so the optimum was not bracketed.
- **The RNNs gain far more from a higher learning rate** (+2.3 to +5.2 BLEU) than the Transformer (+1.2). The 7e-4 default under-trained everything, and the RNNs most.
- **Dropout is second-order for the RNNs.** For the Transformer, more than 0.2 hurts.

**Tuned comparison:** each model's lowest mean validation NLL configuration, run on 5 seeds.

| Model | Selected | BLEU | ΔBLEU [95% CI] | p (Holm) |
|---|---|---|---|---|
| Transformer | dropout 0.2 | 36.68 ± 0.32 | ref | |
| BiLSTM + Luong | LR ×2 | 36.45 ± 0.50 | −0.23 [−1.19, +0.74] | 0.65 |
| BiGRU + Bahdanau | LR ×2 | 36.13 ± 0.27 | −0.55 [−1.48, +0.31] | 0.50 |
| LSTM, no attention | LR ×2 | 17.60 ± 0.49 | −19.08 [−20.29, −17.85] | 0.003 |

**This table does not show parity.** The Transformer's dropout 0.2 beat its LR ×2 by **0.0002 validation NLL** (1.83969 vs 1.83988), far below the seed noise of about 0.007, so the choice was a coin flip. LR ×2 scores 37.95 vs 36.67 test BLEU on the same seeds. Under equal learning rate (×2 for all), the Transformer leads the attention RNNs by about 1.6–1.8 BLEU, with all three seeds separated. That figure is untested: a 3-vs-3 seed permutation test cannot go below p = 0.1, and the 5-seed bootstrap wasn't run.

### 5.4 Stability: BiLSTM + Luong silently collapses

Seed 3 of BiLSTM + Luong reached 13.6 BLEU at LR ×0.5, 15.2 at dropout 0.1 and 22.6 at dropout 0.2. The other seeds of those configurations reached ~33. 13–15 BLEU is the no-attention model's level, which suggests the attention never learned to work in those runs. Nothing crashed, so no failure was recorded. At LR ×2 the same seed trains normally (35.70). Low learning rates make this baseline unreliable.

### 5.5 Checkpoint averaging and beam search: paired on the same runs, 5 seeds

| Model | Averaged − best checkpoint | p (Holm) | Beam 4 − greedy | p (Holm) |
|---|---|---|---|---|
| Transformer | +0.62 [+0.08, +1.19] | 0.084 | +1.15 [+0.37, +1.95] | 0.024 |
| BiGRU + Bahdanau | −0.43 [−1.01, +0.24] | 0.18 | +1.56 [+0.70, +2.39] | 0.008 |
| BiLSTM + Luong | −1.29 [−1.96, −0.63] | 0.012 | +1.08 [+0.47, +1.70] | 0.015 |
| LSTM, no attention | −1.73 [−2.24, −1.27] | 0.008 | +0.49 [+0.00, +0.95] | 0.094 |

- **Beam search helps every model.**
- **Checkpoint averaging is not neutral.** It helps the Transformer slightly (not significant after correction) and hurts two RNNs significantly.
- **The window was too wide:** snapshots every 500 updates, keep 5, covers about the last 60% of a ~3,850-update run.
- **Effect on the headline:** with the best checkpoint, the main-table gaps would be about −1.7 (GRU) and −2.4 (Luong) instead of −2.7 and −4.3, estimated from the paired differences.

## 6. Transformer ablations

Each variant ran on seeds 1–3 against the 5-seed default (36.71 BLEU, validation NLL 1.918). Holm correction is over all 10 ablations; values are mean ± std.

| Variant (paper Table 3 row) | Change | BLEU | ΔBLEU [95% CI] | p (Holm) | Valid NLL | Interpretation |
|---|---|---|---|---|---|---|
| heads1 (A) | 1 head, d_k = 256 | 37.14 ± 0.28 | +0.43 [−0.26, +1.07] | 0.40 | 1.889 | Not significant: head count doesn't matter at this scale |
| heads8 (A) | 8 heads, d_k = 32 | 36.03 ± 0.33 | −0.68 [−1.41, +0.03] | 0.21 | 1.965 | Not significant |
| layers1 (C) | 1+1 layers | 34.48 ± 0.29 | −2.24 [−3.02, −1.45] | 0.010 | 2.079 | **Significant:** depth helps up to 3 layers |
| layers6 (C) | 6+6 layers | 21.66 ± 0.13 | −15.05 [−16.13, −13.89] | 0.010 | 2.609 | **Optimization failure**, not a depth effect (below) |
| no_dropout (D) | dropout 0 | 28.57 ± 0.71 | −8.15 [−9.35, −6.99] | 0.010 | 2.468 | **Significant:** overfits 29k sentence pairs |
| no_label_smoothing (D) | ε = 0 | 36.24 ± 0.49 | −0.47 [−1.20, +0.28] | 0.40 | 1.960 | Not significant |
| learned_pe (E) | learned positions | 33.69 ± 0.37 | −3.02 [−3.85, −2.16] | 0.010 | 2.073 | Significant but **confounded** (below) |
| pre_ln | Pre-LN | 35.62 ± 0.27 | −1.09 [−1.80, −0.40] | 0.010 | 1.979 | **Significant:** Post-LN is better at 3 layers |
| untied | no weight tying | 35.55 ± 0.33 | −1.16 [−1.90, −0.38] | 0.020 | 1.986 | **Significant:** tying regularizes small data |
| no_embed_scale | no √d embedding scale | 13.97 ± 6.14 | −22.74 [−28.81, −17.26] | 0.010 | 2.853 | Significant but **confounded** (below) |

**`layers6` (21.7 BLEU) is an optimization failure, not evidence that depth hurts.**
- The variant changed depth but kept the 3-layer schedule: warmup 400, peak LR 7e-4, about 3,850 updates. The paper used 4,000 warmup steps and 100k updates.
- Its *best* validation NLL (2.61) is far worse than the default's (1.92), and all three seeds agree tightly. That rules out overfitting, which would show a good early optimum followed by decline.
- Deep Post-LN stacks are known to need long warmup and many updates (Xiong et al., 2020; Liu et al., 2020).
- An implementation fault is unlikely: the stacks are parity-tested for N ≥ 2, and 1 → 3 layers behaves sensibly.
- **To confirm:** compare final training NLL and grad-norm spikes in `metrics.jsonl` against `multi30k_tiny_s1`.
- Week 9's 6+6 Post-LN IWSLT model was gated on a pilot for this reason (ADR-023). It also failed at the full 30k-update budget (§2.8).

**Confounded ablations:** these two compare a component *together with* an initialization mismatch.
- **learned_pe:** learned positions are initialized N(0, 1/d), about 16× weaker than the √d-scaled token embeddings, so the variant starts with almost no position signal.
- **no_embed_scale:** without √d, token embeddings (N(0, 1/d)) are about 11× weaker than the sinusoidal encoding.

Neither result is evidence against learned positions or for scaling as such. The paper found learned and sinusoidal positions equivalent.

## 7. Statistical protocol and what is significant

**Methods:**
- **Primary test:** a two-level paired bootstrap (B = 1000). It resamples test sentences, shared by both systems, and each system's seeds. The CI and p therefore include both test-sampling and training variance.
- **BLEU on each resample** comes from sacreBLEU's own sufficient statistics (equal to sacreBLEU, tested).
- **Multiple comparisons:** Holm-Bonferroni within each table.
- **Supplementary:** an exact seed permutation test. Its smallest attainable p is 2/252 ≈ 0.008 for 5-vs-5 seeds, and 1/56 ≈ 0.018 (one-sided extreme) or 2/56 ≈ 0.036 (symmetric) for 3-vs-5 seeds.
- **Floor:** p printed as "≤ 0.001" is the bootstrap floor 1/(B + 1).

**Significant** (Holm < 0.05):
- every Transformer-vs-RNN gap under protocol defaults and under valid-selected decoding;
- the no-attention LSTM's gap even when tuned;
- beam search over greedy for the Transformer, GRU and Luong;
- averaged worse than best for Luong and the no-attention LSTM;
- the ablations layers1, layers6, no_dropout, untied, pre_ln, learned_pe and no_embed_scale (the last three interpreted above).

**Not significant:**
- the tuned gaps to Luong (−0.23) and the GRU (−0.55);
- averaging for the Transformer (+0.62, Holm 0.084) and the GRU;
- beam search for the no-attention LSTM;
- heads1, heads8, no_label_smoothing.

**Not tested** (descriptive, 3 seeds): the training-sensitivity grid and the equal-learning-rate comparison.

**Validation NLL vs test BLEU:**
- Over all 118 runs the correlation is strong: Spearman ρ = −0.964.
- It is not a reliable *selector* between near-optimal configurations: dropout 0.2 and LR ×2 have equal NLL but differ by 1.3 BLEU.

## 8. Limitations and design lessons (Multi30k)

1. **Learning-rate grid too narrow:** the optimum sits at the grid edge for every model. Future grids must bracket it, and the report now flags edge selections.
2. **Selection by validation NLL can produce coin flips.** Selection is now by validation BLEU, with NLL only as a tie-breaker.
3. **The checkpoint-averaging window was too wide and hurt the RNNs.** The checkpoint (best or averaged) is now chosen per model on validation. Averaging windows should cover roughly the last 10–20% of training.
4. **One-factor ablations at fixed hyperparameters cannot test large architectural changes** (layers6) or changes coupled to initialization (learned_pe, no_embed_scale).
5. **A collapsed seed** (attention never learning) is not a crash. The report now flags outlier seeds.
6. **Scale:** Multi30k with 29k sentence pairs and 7.6M parameters says nothing about large-scale behavior. Week 9 (IWSLT14, ADR-023) is the main benchmark.
7. **Single test set** (test 2016); a single language pair and direction.

## 9. Conclusions (Multi30k)

1. At matched size, data, recipe, decoding and scoring, the Transformer outperforms GRU + Bahdanau and BiLSTM + Luong by **2.7–4.3 BLEU** under the protocol defaults. This is significant and robust across 5 seeds and all decoding settings.
2. Much of that margin comes from a shared learning rate that suited the Transformer better. At a learning rate that suits all models, the lead shrinks to **about 1.6–1.8 BLEU** (3 seeds, untested). Tuned-comparison claims of parity are not supported (§5.3).
3. **Attention is the decisive ingredient:** removing it costs 19–24 BLEU, an order of magnitude more than the choice between recurrence and self-attention.
4. **Second-order factors** on this task: decoding settings (±0.5), checkpoint averaging (−1.7 to +0.6, depending on the model), head count, label smoothing.
5. **Clear Transformer findings:** dropout is essential on small data (−8 without it); Post-LN beats Pre-LN at 3 layers; weight tying helps; 3 layers beat 1.
6. The 6-layer result calls for a properly scheduled deep model. That was Week 9's first gate, not a conclusion about depth. At full budget on IWSLT'14, 6+6 Post-LN still failed (§2.8).

---

## 10. Project conclusions

1. **The central claim holds on both benchmarks.** A hand-written, parity-tested Transformer is the best model at matched size, data, training budget, decoding and scoring:
   - IWSLT'14: 33.10 vs 32.04 BLEU over the best RNN;
   - Multi30k: 36.71 vs 33.97.
2. **The margin depends heavily on tuning.** Under shared, untuned defaults the Transformer led by 2.7–4.3 BLEU (Multi30k). With learning rates suited to every model, the lead is about **1–2 BLEU**: 1.06 on IWSLT'14, and 1.6–1.8 on Multi30k (untested).
3. **Attention is the decisive ingredient:** removing it costs 19–24 BLEU on Multi30k, an order of magnitude more than the choice between recurrence and self-attention.
4. **The right normalization placement depends on depth and recipe.** Post-LN beats Pre-LN at 3+3 layers (+1.1 BLEU, Multi30k). At 6+6 on IWSLT'14 it failed to train in this recipe, so the headline model is a declared Pre-LN deviation.
5. **The evaluation protocol caught failures a single number would hide:** a collapsed BiLSTM seed on Multi30k, the GRU's late divergence on IWSLT'14, coin-flip NLL selection, and a harmful averaging window. These were caught by:
   - validation-only selection;
   - reporting protocol defaults next to tuned results;
   - training-health and outlier flags.

## 11. Future work
1. **Post-LN at 6+6:** per-layer gradient norms at initialization, then one-factor runs over Adam ε, weight decay, Q/K/V gain, tokens per update and tying (ADR-023 Amendment 4).
2. **GRU + Bahdanau stability:** a full-budget learning-rate check and the cause of the late divergence.
3. **Decoding:** a wider grid (beam ≥ 8, α > 1.0), since every model selected the grid corner.
4. **Figures** (DESIGN.md §11.3): training curves, attention maps, positional-encoding heatmaps, per-layer gradient norms, BLEU vs GPU-hours.
5. **Scale:** more seeds; WMT'14 EN→DE with the base model.
