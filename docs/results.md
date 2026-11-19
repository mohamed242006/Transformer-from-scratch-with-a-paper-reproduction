# Results

Every number on this page comes from preserved runs in the private `aiayn-runs` dataset (ADR-020). Each can be regenerated from them:
- `scripts/compare_runs.py` produces Week 7's `COMPARISON.md`;
- `scripts/analyze_study.py --study configs/study/week8.yaml` produces Week 8's `STUDY.md` and `STUDY.json`.

**Common setup:**
- **Data:** Multi30k Task 1, EN→DE, test 2016 (1,000 sentences); raw cased text; joint SentencePiece BPE 8k.
- **Training recipe** (`multi30k_t4`): 30 epochs, token-budget batches of 4,096 tokens per side, inverse-sqrt schedule (peak 7e-4, warmup 400), label smoothing 0.1, Adam (0.9, 0.98, 1e-9).
- **Decoding:** beam 4, α = 0.6, max length |x| + 50.
- **Metric:** sacreBLEU, cased and detokenized (`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`).
- **Hardware:** a single Kaggle T4, fp16.
- **Unless stated otherwise:** averaged checkpoint (mean of the last 5 snapshots).

Contents: [1. Headline](#1-headline) · [2. Week 7: baselines](#2-week-7-rnn-baselines-adr-021) · [3. Week 8: study](#3-week-8-multi-seed-study-adr-022) · [4. Ablations](#4-transformer-ablations) · [5. Statistics](#5-statistical-protocol-and-what-is-significant) · [6. Limitations](#6-limitations-and-design-lessons) · [7. Conclusions](#7-conclusions)

---

## 1. Headline

| Model (≈ 7.6M parameters, matched within 1%) | Protocol defaults, test BLEU (5 seeds) | Valid NLL | LR ×2, test BLEU (3 seeds) |
|---|---|---|---|
| Transformer (Post-LN, 3+3 layers, d = 256) | **36.71 ± 0.20** | 1.918 | **37.95** |
| BiGRU + Bahdanau attention | 33.97 ± 0.15 | 2.210 | 36.32 |
| BiLSTM + Luong attention | 32.46 ± 0.51 | 2.355 | 36.16 |
| LSTM encoder-decoder, no attention | 12.60 ± 0.26 | 3.071 | 17.73 |

- **Under identical training, the Transformer is the best model.**
  - Under the protocol defaults, every gap is significant: two-level paired bootstrap, Holm-adjusted p ≈ 0.003.
  - With a learning rate tuned for all models (×2), the gap to the attention RNNs narrows to about **1.6–1.8 BLEU**. That comes from 3 seeds and wasn't formally tested (§3.3).
- **Attention is the decisive component:** without it, the encoder's fixed-size state costs 19–24 BLEU.

## 2. Week 7: RNN baselines (ADR-021)

The baselines use the same data, tokenizer, batching, schedule, epochs, label smoothing, checkpoint averaging, decoding and scoring as the Transformer. The trainer, beam search and evaluation code are literally shared, through the `EncoderDecoder` interface. The one declared difference is `clip_norm = 1.0` for the RNNs. Parameter counts are within 1% of the Transformer's 7,577,600.

Seed-1 results (`COMPARISON.md`, all fairness checks PASS):

| Model | Params | Valid NLL | BLEU | 95% CI (test sampling) | ΔBLEU | p | chrF | Len ratio | Train time | Target tokens/s |
|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | 7,577,600 | 1.910 | **36.87** | ±1.63 | ref | | 60.74 | 0.963 | 4.1 min | 62,797 |
| BiGRU + Bahdanau | 7,513,536 | 2.212 | 34.14 | ±1.58 | −2.73 | ≤ 0.001 | 59.21 | 1.039 | 5.3 min | 47,942 |
| BiLSTM + Luong | 7,580,496 | 2.343 | 32.65 | ±1.60 | −4.22 | ≤ 0.001 | 56.65 | 1.024 | 2.8 min | 99,054 |
| LSTM, no attention | 7,536,384 | 3.072 | 12.28 | ±0.85 | −24.59 | ≤ 0.001 | 35.42 | 1.094 | 2.4 min | 115,076 |

**Speed:** at this tiny scale the cuDNN LSTMs are faster per token than the hand-written Transformer. The Bahdanau decoder is slowest because it loops over time steps in Python. Week 8 confirmed every Week 7 number to within 0.3 BLEU of its 5-seed mean.

## 3. Week 8: multi-seed study (ADR-022)

The study is `configs/study/week8.yaml`: 118 runs, all completed, none failed. Selections were made on validation data only.

### 3.1 Main comparison: protocol defaults, 5 seeds

| Model | BLEU | min–max | chrF | Len ratio | ΔBLEU [95% CI] | p (Holm) | p (seed permutation) |
|---|---|---|---|---|---|---|---|
| Transformer | **36.71 ± 0.20** | 36.49–36.96 | 60.74 | 0.965 | ref | | |
| BiGRU + Bahdanau | 33.97 ± 0.15 | 33.79–34.14 | 59.20 | 1.041 | −2.74 [−3.69, −1.84] | 0.003 | 0.008 |
| BiLSTM + Luong | 32.46 ± 0.51 | 31.59–32.90 | 56.36 | 1.021 | −4.25 [−5.24, −3.22] | 0.003 | 0.008 |
| LSTM, no attention | 12.60 ± 0.26 | 12.28–12.99 | 35.69 | 1.089 | −24.11 [−25.35, −22.78] | 0.003 | 0.008 |

- **Seed-to-seed spread is small:** standard deviation 0.15–0.51 BLEU, so each gap is many standard deviations wide.
- **chrF and validation NLL give the same ranking.**

### 3.2 Decoding sensitivity: selected on validation, reported on test

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

### 3.3 Training sensitivity: one factor at a time, seeds 1–3, mean test BLEU

| Model | LR ×0.5 | default | **LR ×2** | dropout 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|---|
| Transformer (default dropout 0.1) | 33.97 | 36.77 | **37.95** | = default | 36.67 | 35.06 | 30.67 |
| BiLSTM + Luong (default 0.3) | 19.59* | 32.25 | **36.16** | 27.26* | 29.98* | = default | 31.39 |
| BiGRU + Bahdanau (default 0.3) | 28.54 | 34.00 | **36.32** | 34.23 | 34.01 | = default | 33.19 |
| LSTM, no attention (default 0.3) | 9.13 | 12.49 | **17.73** | 13.27 | 13.11 | = default | 12.06 |

\* includes a collapsed seed (§3.4).

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

### 3.4 Stability: BiLSTM + Luong silently collapses

Seed 3 of BiLSTM + Luong reached 13.6 BLEU at LR ×0.5, 15.2 at dropout 0.1 and 22.6 at dropout 0.2. The other seeds of those configurations reached ~33. 13–15 BLEU is the no-attention model's level, which suggests the attention never learned to work in those runs. Nothing crashed, so no failure was recorded. At LR ×2 the same seed trains normally (35.70). Low learning rates make this baseline unreliable.

### 3.5 Checkpoint averaging and beam search: paired on the same runs, 5 seeds

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

## 4. Transformer ablations

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
- Week 9's 6+6 Post-LN IWSLT model is gated on a pilot for this reason (ADR-023).

**Confounded ablations:** these two compare a component *together with* an initialization mismatch.
- **learned_pe:** learned positions are initialized N(0, 1/d), about 16× weaker than the √d-scaled token embeddings, so the variant starts with almost no position signal.
- **no_embed_scale:** without √d, token embeddings (N(0, 1/d)) are about 11× weaker than the sinusoidal encoding.

Neither result is evidence against learned positions or for scaling as such. The paper found learned and sinusoidal positions equivalent.

## 5. Statistical protocol and what is significant

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

## 6. Limitations and design lessons

1. **Learning-rate grid too narrow:** the optimum sits at the grid edge for every model. Future grids must bracket it, and the report now flags edge selections.
2. **Selection by validation NLL can produce coin flips.** Selection is now by validation BLEU, with NLL only as a tie-breaker.
3. **The checkpoint-averaging window was too wide and hurt the RNNs.** The checkpoint (best or averaged) is now chosen per model on validation. Averaging windows should cover roughly the last 10–20% of training.
4. **One-factor ablations at fixed hyperparameters cannot test large architectural changes** (layers6) or changes coupled to initialization (learned_pe, no_embed_scale).
5. **A collapsed seed** (attention never learning) is not a crash. The report now flags outlier seeds.
6. **Scale:** Multi30k with 29k sentence pairs and 7.6M parameters says nothing about large-scale behavior. Week 9 (IWSLT14, ADR-023) is the main benchmark.
7. **Single test set** (test 2016); a single language pair and direction.

## 7. Conclusions

1. At matched size, data, recipe, decoding and scoring, the Transformer outperforms GRU + Bahdanau and BiLSTM + Luong by **2.7–4.3 BLEU** under the protocol defaults. This is significant and robust across 5 seeds and all decoding settings.
2. Much of that margin comes from a shared learning rate that suited the Transformer better. At a learning rate that suits all models, the lead shrinks to **about 1.6–1.8 BLEU** (3 seeds, untested). Tuned-comparison claims of parity are not supported (§3.3).
3. **Attention is the decisive ingredient:** removing it costs 19–24 BLEU, an order of magnitude more than the choice between recurrence and self-attention.
4. **Second-order factors** on this task: decoding settings (±0.5), checkpoint averaging (−1.7 to +0.6, depending on the model), head count, label smoothing.
5. **Clear Transformer findings:** dropout is essential on small data (−8 without it); Post-LN beats Pre-LN at 3 layers; weight tying helps; 3 layers beat 1.
6. The 6-layer result calls for a properly scheduled deep model. That is Week 9's first gate, not a conclusion about depth.
