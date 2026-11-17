# ADR-021: RNN baselines and the fair-comparison protocol

- **Status:** Accepted
- **Date:** 2026-11-17
- **Deciders:** Lead engineer, project owner
- **Related:** [ADR-001](ADR-001-manual-implementation.md), [ADR-006](ADR-006-evaluation-protocol.md), [ADR-019](ADR-019-decoding-and-evaluation.md), [ADR-020](ADR-020-kaggle-run-persistence.md), [DESIGN.md §7](../DESIGN.md)

## Context
Week 6 established the Transformer reference on Multi30k EN→DE: `multi30k_tiny_s1`, averaged checkpoint, beam 4, α = 0.6. Test results: **BLEU 36.87, chrF 60.74, length ratio 0.963**, 7,577,600 parameters. A claim that "the Transformer is better" is only meaningful if the baselines are trained, decoded and scored the same way. Two baselines given a separate training script, a different tokenizer or their own beam search would make the comparison unfalsifiable.

## Decision
1. **One interface, one pipeline.** `aiayn.models.EncoderDecoder` defines `encode`, `decode`, `generator` and `max_positions`. The Transformer and `RNNSeq2Seq` both implement it. The Trainer, beam and greedy search, the Translator, `evaluate_run` and the reports have no architecture branches. `build_model` dispatches on the config: `architecture: rnn` gives an `RNNConfig`, anything else a `ModelConfig`. Transformer configs carry no new key, so their config hash is unchanged and existing runs stay resumable.
2. **Three baselines** (DESIGN.md §7), all points in one configurable class:

   | Config | Encoder | Decoder | Attention | Params (V = 8k) | vs Transformer |
   |---|---|---|---|---|---|
   | `lstm_seq2seq` | LSTM 2×432 | LSTM 2×432 | none (Sutskever et al., 2014) | 7,536,384 | −0.5% |
   | `gru_bahdanau` | BiGRU 1×448 | GRU 1×448 | additive (Bahdanau et al., 2015) | 7,513,536 | −0.8% |
   | `bilstm_luong` | BiLSTM 2×296 | LSTM 2×296 | global "general" (Luong et al., 2015) | 7,580,496 | +0.04% |

   The following are shared with the Transformer:
   - embedding width 256, joint vocabulary, three-way weight tying, no output bias;
   - a tanh readout projects to the embedding width before the tied softmax.

   Other architecture details:
   - Bidirectional final states reach the decoder through `tanh(W[fwd; bwd])`.
   - Initialization is uniform(−0.1, 0.1).
   - Dropout is 0.3 on embeddings, between layers, on encoder outputs and on the readout.
   - `nn.LSTM`/`nn.GRU` are allowed in `models/baselines/` only (ADR-001; a policy test enforces this). The attention is hand-written, with its softmax in ≥ fp32.
3. **Deliberate simplifications:**
   - no source reversal (Sutskever);
   - no input feeding (Luong), which keeps teacher-forced training parallel over time;
   - a tanh readout instead of maxout (Bahdanau).

   Each would add roughly 0–1 BLEU in the original papers, so none changes the conclusion. They are recorded here so the baselines are not over-claimed.
4. **Fairness protocol** (DESIGN.md §7.3), checked by `aiayn.inference.comparison` from the run files (not assumed):
   - same data config, tokenizer fingerprint and reference lines;
   - same decoding (beam 4, α 0.6, max length |x| + 50) and checkpoint choice (averaged);
   - parameters within ±10% (the configs are within 1%);
   - the same training config: `multi30k_t4` with 30 epochs, token budget, inverse-sqrt schedule, label smoothing 0.1, Adam and checkpoint averaging;
   - every training difference listed, and anything not declared marked as a failure.
5. **The one declared difference:** RNN experiments set `train.clip_norm=1.0`. Gradient clipping is the standard RNN stabilizer. It can only help the baselines, so the comparison stays conservative against the Transformer. Dropout 0.3 (vs 0.1) is an architecture hyperparameter chosen in the baselines' favour on a small corpus for the same reason.
6. **Significance:** paired bootstrap resampling (Koehn, 2004; sacreBLEU `PairedTest`, 1000 samples) of each system against the reference, reporting the 95% CI and p-value. With several seeds per experiment, mean ± std is added. Single-seed results are reported as such.
7. **Decoding cost:** every model recomputes the decoder over the prefix at each step, with no KV cache and no carried RNN state. This is exact because the decoders are causal (tested), but decode time compares implementations rather than architectures. Incremental decoding is an extension.

## Consequences
**Positive:**
- A baseline cannot silently get different data, decoding or scoring.
- One Kaggle session trains all baselines and writes `COMPARISON.md`.
- Bit-identical resume holds for the RNNs too (tested).

**Negative:**
- `RNNConfig.max_positions` exists only as a decode-length cap.
- The Bahdanau decoder is a Python loop over time; it is slower, but fine at Multi30k scale.

**Open (Week 8):**
- the LR sweep (identical grid for every model);
- 3 seeds;
- IWSLT14 source pinning, now ADR-022.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Separate RNN training/decoding scripts | Exactly the unfairness this ADR prevents |
| `architecture` field on `ModelConfig` | Changes every Transformer config hash; existing runs could not resume |
| Hand-written LSTM/GRU cells | Not the subject of the reproduction (ADR-001); cuDNN kernels keep the baselines' throughput honest |
| Luong input feeding | Forces a sequential decoder; small gain (≈ 1 BLEU in the paper), documented instead |
| Identical clip settings for all (none) | Risk of RNN divergence would make the baselines artificially weak |

## Compliance
- `tests/unit/test_rnn_baselines.py` covers:
  - analytic parameter counts and ±10% sizing;
  - causality, padding invariance and beam expansion;
  - attention distributions, masking and gradcheck;
  - beam 1 equals greedy through the shared decoders;
  - overfitting one batch.
- `tests/integration/test_baselines_pipeline.py` covers:
  - Trainer with bit-identical stop/resume;
  - `evaluate_run`;
  - the comparison's fairness checks, including detecting an undeclared difference.
- `tests/policy/test_forbidden_apis.py` confines recurrent modules to `models/baselines/`.
