# Experiment Log

Chronological lab notebook. **Append only.** Record failures and dead ends as carefully as successes; they are the most useful part of the record.

Entry format: date · week · what was done · findings · decisions · next steps.

---

## 2026-11-15 · Week 1 · Paper study and repository architecture

**Done**
- Read *Attention Is All You Need* (arXiv v7) in full. Transcribed and checked Tables 1–4 and every training/inference hyperparameter → [paper_notes.md](paper_notes.md).
- Derived every equation the implementation depends on → [math.md](math.md).
- Wrote ADR-001…005 → [decisions/](decisions/README.md).
- Defined the repository architecture, ownership and import layering → [REPOSITORY.md](REPOSITORY.md).
- Created the package skeleton (`aiayn`, docstrings only), tooling (ruff, mypy, pytest, pre-commit, CI) and the ADR-001 policy test.

**Findings from the paper review**
1. EN→FR BLEU is reported as **41.8** (abstract, Table 2) and **41.0** (§6.1): an internal inconsistency.
2. The analytic parameter count for the base model with V = 37,000 is **63.08M** vs the reported **65M**. Table 3 rows imply an effective vocabulary of about 36–41k.
3. **Attention dropout** is never defined in §3 or §5, but is mentioned in §6.3 ("dropout, both attention and residual").
4. §5.4 announces "three types of regularization" but describes two.
5. Paper cost estimate (3.3·10¹⁸ FLOPs) vs the 6·P·T estimate (≈ 9.5·10¹⁷) implies about **29%** hardware utilization.

**Spec amendments adopted in Week 1 (spec v1.0 → v1.1)**
| # | Amendment | Reason |
|---|---|---|
| A1 | Embeddings initialized N(0, d^-½), not Xavier | Required for √d scaling + tying to produce unit-variance inputs and logits (math.md §5) |
| A2 | Overfit-one-batch tests use ε_ls = 0 | Smoothed loss floor H(q′) ≈ 1.2–1.4 nats (math.md §8.4) |
| A3 | Initial-loss check is a band [ln V, ln V + 1.5] | Expected ≈ ln V + 0.5 (math.md §5.3) |
| A4 | Label smoothing convention: ε/K over the full vocabulary | Allows `F.cross_entropy` as the parity reference (math.md §8.1) |
| A5 | Package renamed `transformer` → `aiayn` | One character from HF `transformers`; risk of import typos |
| A6 | Added gitignored `data/` and `docs/figures/` | Missing from the spec tree |
| A7 | Masks live in `modules/`, not `utils/` | Masks define model semantics and are parity-tested |
| A8 | Added `tests/parity/` and `tests/policy/` | ADR-004 and ADR-001 enforcement |
| A9 | Dual BLEU reporting for IWSLT14 | Published reference numbers use lowercased, tokenized BLEU (ADR-002) |
| A10 | Checkpoints written by step count, not wall-clock | Reproducible across hardware |

**Next (Week 2)**
- ADR-006 evaluation protocol; ADR-007 configuration system.
- Pin the IWSLT14 source + checksums; Multi30k and copy-task data; SentencePiece training; token-budget sampler; data tests.

---

## 2026-11-15 · Week 2 · Data pipeline

**Pre-flight review fixes:** `docs/DESIGN.md` created (spec v1.1; it previously existed only in conversation) · ruff pinned to 0.17.1 in pyproject, pre-commit and CI · `types-PyYAML` added · relative imports banned (TID252) · layering + "sentencepiece only in `aiayn.data`" policy test · ADR-006…009 written; ADR-010 (compute platform) **proposed**.

**Done**
- Config system (ADR-007): strict YAML → frozen dataclasses, dotted overrides, repo-root path resolution.
- Data: special tokens, framing + teacher forcing (ADR-009), copy/reverse task, SentencePiece wrapper + trainer, parallel-text dataset with flat int32 storage and filter stats, token-budget bucketed sampler, collator/`Batch`, checksum-verified downloader, config-driven factory.
- Masks (ADR-008): padding, causal, decoder masks plus an invariant validator, in `aiayn.modules.masking`.
- Scripts: `prepare_data.py`, `train_tokenizer.py`, `inspect_data.py`.

**Results (real data, laptop CPU)**
| Check | Result |
|---|---|
| Multi30k download | 6/6 files SHA-256 verified; 29,000 / 1,014 / 1,000 aligned pairs |
| Tokenizer | BPE 8,000; exact round-trip; fingerprint `f04bec59…` |
| Filtering (max_len 100) | 0 pairs dropped on every split |
| Lengths (train) | S mean 15.3 / max 49; T mean 15.8 / max 53 |
| Batching (train, 4,096 tokens) | 128 batches, padding efficiency 0.950; full pass 0.27 s |
| Copy task (train, 2,048 tokens) | 140 batches, padding efficiency 0.970 |

**Findings**
1. **SentencePiece model files are not reproducible byte-for-byte:** the trainer embeds its arguments (including the output path) in the `.model` file. The tokenizer fingerprint is therefore a *content* hash (pieces, scores, types + a normalization probe), so identical tokenizers match across machines (ADR-005's "referenced by hash" intent; it no longer means the file hash).
2. The file-writing tool converts `\uXXXX` escapes into literal characters. Invisible characters in source are now built with `chr()`, and a scan for non-ASCII characters was added to the review checklist.
3. Hardware audit: **no CUDA GPU** (Iris Xe, 7.6 GB RAM) → ADR-010 proposed (cloud GPU for training, resumable checkpoints).

**Next (Week 3):** awaiting ADR-010 approval · ADR-011 initialization · ADR-012 dropout locations · modules + parity tests.

---

## 2026-11-15 · Week 3 · Core modules

**Pre-flight (Kaggle as the target platform, ADR-010 accepted)**
- New ADRs: **011** initialization, **012** dropout locations (paper text first, else 0.0, which supersedes Week 1's provisional attention dropout of 0.1), **013** run artifacts/checkpointing/resumption on Kaggle, **014** precision policy (fp16 + GradScaler on T4, fp32 reductions, no bf16 on Kaggle GPUs).
- Week 2 code adapted: `TokenBudgetBatchSampler.set_epoch(epoch, start_batch)` + `state_dict()/load_state_dict()` for exact mid-epoch resumption; `prepare_data.py --mirror-dir` for offline, checksum-verified data on Kaggle; CI matrix includes Python 3.11 (Kaggle image).

**Done:** `TokenEmbedding`, `SinusoidalPositionalEncoding`/`LearnedPositionalEmbedding`, `scaled_dot_product_attention`, `MultiHeadAttention` (+ `split_heads`/`merge_heads`), hand-written `LayerNorm`, `PositionwiseFeedForward`, `SublayerConnection` (post/pre), ADR-011 initialization helpers.

**Verification**
| Check | Result |
|---|---|
| Parity vs `nn.LayerNorm`, `F.scaled_dot_product_attention`, `nn.MultiheadAttention` | Pass at ADR-004 tolerances (fp64 1e-10 / fp32 1e-6), outputs + per-head weights + input and parameter gradients |
| float64 gradcheck | Attention, LayerNorm (input + γ/β), MHA, FFN, Add & Norm (post and pre) |
| Low precision | fp16/bf16 attention and LayerNorm finite and within tolerance of fp32; masked weights exactly 0; a naive fp16 variance overflows on the same input while ours does not |
| Suite | 280 passed, 1 skipped (CUDA pinning); coverage 93% (lines + branches) |

**Findings**
1. **A helper named `_apply` shadowed `nn.Module._apply`**, which `.to()`, `.double()` and `.cuda()` use. Moving a `SublayerConnection` to a GPU on Kaggle would have crashed. Caught by gradcheck (`.double()`), fixed, with a regression test that moves every module across dtypes.
2. Constructing an `nn.Linear` consumes random numbers before our `reset_parameters()`. Determinism therefore holds per (seed, construction sequence), which matters for Week 4's model-level seeding.
3. All parity tests passed at the fixed ADR-004 tolerances on the first run; no tolerance was loosened.

**Next (Week 4):** ADR-015 model assembly (tying, stack-level norms, generator without bias) · encoder/decoder layers and stacks · full model · full-stack parity · overfit and initial-loss tests.

---

## 2026-11-15 · Week 4 · Full model

**Pre-flight decisions → [ADR-015](decisions/ADR-015-model-assembly.md):** encoder/decoder as pure stacks (mirroring `nn.TransformerEncoder/Decoder`) · independently constructed layers (no deepcopy) · three-way tying registered once (`embeddings.shared.weight`), output projection without bias · attention weights returned by layers, collected (detached) only via `forward_with_attention` · `reset_parameters()` never applies `nn.Linear` defaults · seeded builds under `fork_rng` · `debug_checks` flag.

**Done:** `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder` (modules); `ModelConfig` + `expected_parameter_count`, `Transformer`, `build_transformer` (models); `count_parameters`/`parameter_breakdown` (utils); `configs/model/{tiny,iwslt,base}.yaml`; `scripts/count_params.py`; Kaggle validation notebook design.

**Parameter counts (counted == analytic == math.md §9.4)**
| Config | V | Embedding | Encoder | Decoder | Total |
|---|---|---|---|---|---|
| tiny | 8,000 | 2,048,000 | 2,369,280 | 3,160,320 | **7,577,600** |
| iwslt (mid) | 10,000 | 5,120,000 | 12,616,704 | 18,926,592 | **36,663,296** |
| base | 37,000 | 18,944,000 | 18,914,304 | 25,224,192 | **63,082,496** |

**DESIGN.md compliance check**
| DESIGN item | Status |
|---|---|
| §2.1 every component hand-written; §2.2 forbidden APIs absent | ✅ (policy test) |
| §3 Post-LN, √d scaling, three-way tying, residual + embedding dropout | ✅ (parity vs PyTorch + unit tests) |
| §4 shapes: src `[B,S]`, tgt_in `[B,T]` → logits `[B,T,V]`; masks `[B,1,1,S]` / `[B,1,T,T]` | ✅ |
| §6.2 tiny / mid / base dimensions and dropout | ✅ (configs) and exact counts |
| §10 parity, invariance, initial loss, save/load tests | ✅ |
| Deviations | Output projection without bias; attention/activation dropout 0.0 (ADR-012); PE module per side; `max_positions` 1024; added `forward_with_attention` API. All recorded in ADR-015. |

**Findings**
1. **The sinusoid table was stored in float32,** so `model.double()` could not recover the lost precision. Every float64 parity and gradcheck computation carried ~1e-7 error. The full-model parity test caught it (layer-level tests passed because they bypass PE). Fixed: the table is stored in float64 and cast per forward; regression test added.
2. **PAD-row gradient semantics:** with `padding_idx`, the PAD row of the tied matrix receives gradient only from the output projection, never from input lookups. The full-model parity reference and the gradcheck were adjusted to state this explicitly; it is the intended behaviour.
3. Layer and stack parity (Post-LN and Pre-LN, fp64 and fp32, forward and backward) passed on the first run. PyTorch's references were built without stack norms for Post-LN, as ADR-003 requires.

**Next (Week 5):** ADR-016 loss · ADR-017 experiment composition · trainer with ADR-013 resumability · Noam schedule · implement the Kaggle validation notebook.

---

## 2026-11-15 · Pre-Week-5 · Kaggle validation workflow

**Done:** `aiayn.diagnostics` (validation checks, JSON report, packaging) · `scripts/kaggle_validate.py` · `scripts/package_for_kaggle.py` · `notebooks/kaggle_validate.ipynb` · [KAGGLE.md](KAGGLE.md) · [ADR-016](decisions/ADR-016-kaggle-code-and-data-delivery.md) (code and data as private Kaggle Datasets with a content tree hash, since the project has no git repository).

**Local results (CPU, real Multi30k):** all critical checks pass. Tokenizer fingerprint `f04bec59…` matches Week 2, round-trip 200/200, and a retrained tokenizer is identical. Exact parameter counts (tiny 7,577,600; iwslt 35,639,296 at V = 8k). Initial CE inside the ln V band. VRAM estimates for the worst-case step at 4096 tokens/side: tiny ≈ 1.1 GiB, iwslt ≈ 3.5 GiB (KAGGLE.md §5). **Kaggle dry run** (code and data run from the packaged bundles, internet check off): 93 preflight tests pass, validation OK in 39 s.

**Findings**
1. **Packaging bug caught by the dry run:** the exclusion pattern `data` matched every path component, silently dropping the `aiayn.data` package and `configs/data/` from the code bundle. Fixed by anchoring root-only excludes; a regression test now checks that every module, config, script and test is shipped.
2. Running the VRAM estimate's worst-case batch with bf16 autocast on a CPU without native bf16 took > 10 min. Activations are linear in the number of sentences at a fixed length, so the estimate now measures a 2-sentence slice and scales it. On GPU the peak is measured directly.
3. Walking `.venv` made packaging slow (7–10 s); directory pruning brought it to 0.04 s.
4. The file-writing toolchain again turned an escape (`\n`) inside generated notebook code into a real newline. The generator now syntax-checks every code cell; the faulty cell was rewritten without escapes.

**Next:** run the notebook on a Kaggle T4 (owner) → record measured VRAM and throughput here → Week 5.

---

## 2026-11-16 · Kaggle validation results + Week 5 · Training system

**Kaggle validation (owner's run, Tesla T4, Python 3.13):** all checks passed except `env.python`. Analysis: the validator hard-coded Python 3.10–3.12 (the CI matrix at the time), while `pyproject.toml` requires only ≥ 3.10 and the preflight test suites had passed on 3.13. **It was a validator assumption, not a real incompatibility.** Fix: fail below 3.10, pass for CI-tested 3.10–3.13, warn for newer; CI matrix extended to 3.13.

| Model | Peak VRAM (worst-case step, 4096 tok/side) | CPU-based estimate | Throughput (fp16, real batches) |
|---|---|---|---|
| transformer_tiny | **898 MiB** | 1,084 MiB | **78,808 tgt tok/s** |
| transformer_iwslt | **2,243 MiB** | 3,618 MiB | **24,669 tgt tok/s** |

Training settings and time estimates derived from these numbers are in [KAGGLE.md §7](KAGGLE.md).

**Week 5 done:** label-smoothed loss (ADR-017) · Noam / inverse-sqrt / constant schedules · `TrainConfig` + experiment composition and config hash (ADR-018) · trainer with fp16 autocast + GradScaler (skipped updates not counted), gradient accumulation with per-update token normalization, gradient-norm logging/clipping, non-finite guard, validation (loss/NLL/perplexity/accuracy), best checkpoint, averaging snapshots, session time budget, SIGTERM/SIGINT clean stop · atomic checkpoints with `last.prev.pt` fallback, `weights_only` loading, RNG capture · stdout / JSONL / TensorBoard / W&B loggers · `scripts/train.py` · `notebooks/kaggle_train.ipynb` · `kaggle_validate.py --check-training-resume`.

**Findings**
1. **`DataLoader` consumes the global torch RNG on every iterator creation**, even with zero workers. A resumed run would therefore shift every later dropout mask. Fix: dedicated `generator=` per loader (`build_dataloader(..., generator=...)`). Verified by bit-identical resumption tests.
2. **Resumption is bit-identical on CPU,** including a session budget that stops after *every single update* (12+ resumes): per-update losses and final weights equal the uninterrupted run exactly.
3. `torch.__version__` is a `TorchVersion` object, so a checkpoint containing it could not be loaded with `weights_only=True`. Caught by the first resume smoke test; stored as `str`.
4. YAML 1.1 parses `amp: off` as the boolean `False`, which the strict config loader rejected. The mode is renamed `fp32`.
5. The loss converted logits with `.float()`, which **downcasts float64** (the same bug class as Week 4's sinusoid table). Now uses `promote_types(dtype, float32)`; regression test added.
6. Resuming in a fresh Kaggle session needs the data re-prepared (the clone has no `data/`). `train.py --resume --prepare-data` downloads and verifies the data but keeps the run's own tokenizer.

**Week 5 exit criterion: copy task** (`copy_task_small`, laptop CPU, 1,500 updates, ≈ 22 min with the CPU shared by the test suite):

| Update | 250 | 500 | 750 | 1,000 | 1,250 | 1,500 |
|---|---|---|---|---|---|---|
| Held-out token accuracy | 74.3% | 98.2% | 99.6% | **99.99%** | 99.7% | 99.9% |

The validation smoothed loss plateaus at ≈ 0.60, the theoretical floor H(q′) ≈ 0.594 for K = 20, ε = 0.1 (math.md §8.4): an independent check of the loss implementation. Sequence-level exact match needs greedy decoding (Week 6).

**Tests:** 444 passed, 1 skipped (CUDA-only), coverage 89%; ruff and strict mypy clean.

**Next (Week 6):** greedy and beam search (GNMT length penalty), checkpoint averaging, sacreBLEU/chrF evaluation, a first Multi30k run on Kaggle.

---

## 2026-11-16 · Multi30k T4 run analysis + Week 6 · Decoding and evaluation

**Kaggle run `multi30k_tiny` (owner, Tesla T4, fp16, 30 epochs, 3,854 updates):** best validation NLL **1.910 (ppl 6.75)** at update 3,469 (epoch 27); token accuracy 64.5%. The schedule was exactly as configured: peak 7.0e-4 at update 400, then 7e-4·√(400/step). Stable: 0 skipped fp16 steps, loss scale grew 2^16 → 2^17 at update 2,000, gradient norm mostly 0.6–1.5 with brief spikes (max 8.96), memory flat at 923 MiB, steady-state ≈ 62–71k target tokens/s. Validation plateaued after about epoch 20, and the train/valid NLL gap grew from ≈ 0.2 (epoch 10) to ≈ 0.86 (epoch 30): **overfitting is starting, but validation is flat, not rising.** Remedies on the plan: checkpoint averaging, dropout 0.2–0.3 ablation, BLEU-based selection.

**Logging bug found in that log:** intervals spanning an epoch end reported 33–48k tok/s because validation and checkpoint time were inside the throughput timer. Fixed: that time is now excluded.

**Week 6 done:** greedy + batched beam search with GNMT length penalty, forced EOS at input + 50, never PAD/BOS (ADR-019) · checkpoint averaging · `Translator.from_run` (best/last/averaged/path) · `evaluate_run` + `scripts/evaluate.py`: cased sacreBLEU with signature, lowercased BLEU, chrF, exact match, BLEU by length · `scripts/translate.py`, `scripts/average_checkpoints.py` · evaluation cell in `kaggle_train.ipynb` · layering: `inference` may import `evaluation`; `sacrebleu` only in `evaluation`.

**Copy-task test set (1,000 sentences, CPU, Week 5 run):**
| Checkpoint | Search | Exact match | BLEU |
|---|---|---|---|
| best | greedy | 99.3% | 99.94 |
| best | beam 4 (first version: stop after k finished) | 93.7% | 98.44 |
| best | **beam 4 (exact stopping rule)** | **99.4%** | **99.95** |
| averaged (500/1000/1500) | beam 4 | 97.1% | 99.65 |

**Findings**
1. **Beam-search stopping bug:** stopping a sentence once *k* hypotheses had finished truncated outputs (all 63 errors were too short). Short EOS candidates from the strong path kept ranking within the top *k*, because the alternatives were even worse. Fixed with an exact rule: stop when the best finished normalized score ≥ `best_live / lp(limit)`. Beam search now matches brute-force exhaustive search (α ∈ {0, 0.6, 1}) and equals greedy at k = 1, α = 0.
2. **Averaging across the whole run hurts** when early snapshots are under-trained (97.1% vs 99.4%). Average plateau checkpoints only. The Multi30k config's last-5 × 500 updates covers the plateau.

**Next:** run the evaluation on the Kaggle Multi30k run (owner) → first real BLEU. Then Week 7 (RNN baselines, IWSLT14 source).

---

## 2026-11-16 · Lost Kaggle run and run-persistence workflow

**What happened:** the Multi30k T4 run (`20261116-175603_multi30k_tiny_s1`) completed, but its run directory was lost: the interactive Kaggle session ended before a version was saved. Only the console log survived (analysis recorded above). Without weights, no BLEU.

**Decision: re-run, don't skip.** The re-run costs ≈ 4–5 min of training + 1–2 min of evaluation on a T4 (from the measured 67k tok/s), under 1% of a weekly quota. It produces the first real-data BLEU (Week 6's open exit criterion) and the Transformer reference that Week 7's RNN baselines are compared against. Same seed and config, so its best validation NLL should land close to the lost run's 1.910, which doubles as a reproducibility check. Skipping to Week 7 would leave decoding and evaluation unverified on real text.

**Workflow change (ADR-020):** commit-mode training notebook (train → evaluate → `REPORT.md` → publish) · private `aiayn-runs` dataset updated each session via the Kaggle CLI (notebook secrets) · stable `RUN_NAME` + auto-resume from any attached input (zips included) · `scripts/report.py`, `scripts/kaggle_publish_runs.py`, `aiayn.diagnostics.kaggle_runs`, `aiayn.inference.report`.

---

## 2026-11-17 · Week 6 closed: first Transformer benchmark · Week 7 baselines

**Reference result** (`multi30k_tiny_s1`, re-run under the ADR-020 workflow), Multi30k test 2016, EN→DE:

| Checkpoint | Search | BLEU | chrF | Length ratio |
|---|---|---|---|---|
| averaged (last 5) | beam 4, α 0.6 | **36.87** | 60.74 | 0.963 |

Week 6 exit criterion met: first real BLEU. The length ratio of 0.963 means slightly short outputs (brevity penalty ≈ 0.96). The α sweep is a Week 8/9 candidate.

**Week 7: baselines (ADR-021).** LSTM seq2seq (no attention), BiGRU + Bahdanau and BiLSTM + Luong. All three are sized within 1% of the Transformer's 7.58M parameters and run through the same trainer, beam search and evaluation via the new `EncoderDecoder` interface. The only declared training difference is `clip_norm=1.0` for the RNNs. `scripts/compare_runs.py` checks fairness from the run files and adds paired-bootstrap significance.

**Next action:** run `notebooks/kaggle_train.ipynb` (default `RUNS`, `aiayn-runs` attached) as Save & Run All (Commit), then record `COMPARISON.md` here.

**Expected:** Transformer > BiLSTM+Luong ≳ GRU+Bahdanau ≫ LSTM without attention. A baseline beating the Transformer on Multi30k would not be implausible at this scale and would be reported as such.

---

## 2026-11-17 · Week 7 results · Week 8 study plan

**Week 7 comparison** (`COMPARISON.md`; test 2016, averaged checkpoint, beam 4, α 0.6, seed 1; all fairness checks PASS):

| Model | Params | Valid NLL | BLEU (95% CI) | ΔBLEU | p | chrF | Len ratio | Wall (min) | tgt tok/s |
|---|---|---|---|---|---|---|---|---|---|
| Transformer 3+3, d=256 | 7,577,600 | 1.910 | **36.87** (±1.63) | ref | | 60.74 | 0.963 | 4.1 | 62,797 |
| BiGRU + Bahdanau | 7,513,536 | 2.212 | 34.14 (±1.58) | −2.73 | ≤ 0.001 | 59.21 | 1.039 | 5.3 | 47,942 |
| BiLSTM + Luong | 7,580,496 | 2.343 | 32.65 (±1.60) | −4.22 | ≤ 0.001 | 56.65 | 1.024 | 2.8 | 99,054 |
| LSTM, no attention | 7,536,384 | 3.072 | 12.28 (±0.85) | −24.59 | ≤ 0.001 | 35.42 | 1.094 | 2.4 | 115,076 |

(`COMPARISON.md` printed p = 0.0010. That value is the floor of a 1000-sample bootstrap, so the formatter now prints "≤0.001".)

**Reading:**
- **Attention is the dominant factor.** Without it, the fixed-size encoder state costs 20+ BLEU.
- **The Transformer leads the best RNN by 2.7 BLEU** at matched size, data and decoding. Its lower validation NLL (1.910 vs 2.212) agrees with the BLEU ordering.
- **Bahdanau beat Luong,** against expectation. Plausible causes are Luong without input feeding (ADR-021 §3) and the smaller hidden size Luong needs at matched parameters (296 vs 448). Week 8's sensitivity grid tests the hyperparameter side.
- **Output lengths differ systematically:** RNNs run long (1.02–1.09), the Transformer short (0.963). One shared α = 0.6 favours neither, which is why Week 8 selects decoding per model on valid.
- **Speed:** at this tiny scale the cuDNN LSTMs are faster per token than the hand-written Transformer (99–115k vs 63k tok/s). The Bahdanau decoder is slowest because it runs a Python loop over time steps. The Transformer's parallelism advantage is expected to appear at larger scale (not tested here).
- **Caveat:** single seed. The ±1.6 CIs cover test sampling only. Week 8 adds training variance.

**Week 8 plan (ADR-022):** `configs/study/week8.yaml`, about 110 runs and about 9 T4-hours:
- main comparison on 5 seeds;
- valid-selected decoding (beam × α grid);
- training sensitivity (LR ×0.5/×2, dropout 0.1–0.4) with valid-NLL selection of a tuned config;
- 10 Transformer ablations (Table 3 style).

Statistics: two-level (sentences × seeds) paired bootstrap with Holm correction, plus seed permutation tests. The roadmap changed: IWSLT14 moves to Week 9 (ADR-023, planned).

**Next action:** run `notebooks/kaggle_study.ipynb` with `aiayn-runs` attached until `STUDY.md` reports *complete*, then turn it into `docs/results.md`.

---

## 2026-11-18 · Week 8 closed: multi-seed study results

`configs/study/week8.yaml`: 118 runs, all completed, none failed. The full write-up is in **[results.md](results.md)**; `STUDY.json` lives in `aiayn-runs/multi30k_tiny_s1/`.

**Key numbers** (Multi30k test 2016, sacreBLEU):

| | Transformer | BiGRU + Bahdanau | BiLSTM + Luong | LSTM, no attention |
|---|---|---|---|---|
| Defaults, 5 seeds | **36.71 ± 0.20** | 33.97 ± 0.15 (−2.74*) | 32.46 ± 0.51 (−4.25*) | 12.60 ± 0.26 (−24.1*) |
| LR ×2, 3 seeds | **37.95** | 36.32 | 36.16 | 17.73 |
| "Tuned" (valid NLL), 5 seeds | 36.68 (dropout 0.2) | 36.13 (−0.55, n.s.) | 36.45 (−0.23, n.s.) | 17.60 (−19.1*) |

\* Holm-adjusted two-level bootstrap p ≈ 0.003.

**Conclusions:**
1. The Transformer leads significantly under the protocol defaults.
2. At a learning rate that suits every model, the lead shrinks to about 1.6–1.8 BLEU (untested).
3. The "tuned" non-significance comes from a 0.0002-nat NLL tie in the Transformer's selection, so it is not evidence of parity.
4. Attention is decisive: −19 to −24 BLEU without it.
5. Learning rate dominates and its optimum lies at the grid edge for every model.
6. Decoding tuning changes test BLEU by ≤ 0.5 and doesn't transfer reliably from validation to test.
7. A wide checkpoint-averaging window hurts the RNNs.
8. Ablations: dropout −8.2; 1 layer −2.2; Pre-LN −1.1; untied −1.2; heads and label smoothing not significant.
9. layers6 (21.7) is an optimization failure under the 3-layer schedule.
10. learned_pe and no_embed_scale are confounded with initialization scale.

**Corrections made:** ADR-022's 3-vs-5 permutation floor is 1/56 ≈ 0.018, not 0.036.

**Adopted for Week 9 (ADR-023):**
- selection by validation BLEU;
- grid-edge flags;
- per-model checkpoint selection;
- outlier-seed flags;
- a training-health table;
- incremental decoding;
- a 6+6 Post-LN pilot gate.

---

## 2026-11-19 · Week 9 Phase 0: IWSLT'14 preparation (no GPU)

**Data** (ADR-023):
- **Source:** fairseq's `de-en.tgz`, SHA-256 `2c0e4d19…f275`, splits built with the `prepare-iwslt14.sh` procedure on raw cased text.
- **Counts:** test **6,750**, identical to fairseq. Train/valid 160,215 / 7,282 vs fairseq's 160,239 / 7,283: 25 pairs, attributed to `sacremoses` vs `tokenizer.perl` near the cleaning thresholds. Counts and output hashes are pinned.
- **Tokenizer:** 10k BPE, fingerprint `b0c6c22a…1989`, deterministic (retrained twice).

**Metrics:** `bleu_fairseq` (Moses-tokenized, lowercased, `tok=none`, `smooth=none`) alongside sacreBLEU for every translation evaluation.

**Decoding:** incremental for all models. Identical hypotheses to prefix decoding (tested); 5.1× (Transformer) and 9.5× (Bahdanau) faster at 40 tokens on CPU.

**Study rules:**
- selection by validation BLEU with NLL tie-break;
- per-model checkpoint choice;
- per-family variant grids;
- grid-edge flags, training-health table, outlier seeds.

Week 8's config pins its old rules.

**Pilot prepared** (`configs/study/week9_pilot.yaml`):
- 10 runs of 6k updates: Transformer at LR {5e-4, 7e-4, 1e-3} plus a Pre-LN arm; GRU + Bahdanau and BiLSTM + Luong at {7e-4, 1.4e-3, 2.8e-3}.
- RNNs resized to 36.57M / 36.61M parameters (vs 36.66M).
- Gate G1–G4 in ADR-023 §8.

**Local smoke test:** the real pipeline (data, pinned tokenizer, training) ran end to end on CPU with a shrunken model.

**Not run:** any Kaggle job (awaiting approval for Phase 1).

---

## 2026-11-19 · Week 9 Phase 1: pilot tooling ready, Kaggle runs pending

**Added:** `aiayn.study.pilot` + `scripts/pilot_report.py`, run by study-notebook cell 5b. They read the pilot runs and the Kaggle validation report and write `PILOT.md` / `PILOT.json` / `pilot_curves.csv`:
- validation curves;
- per-run throughput, peak memory, skipped fp16 steps and maximum gradient norm;
- minutes per 1,000 updates and the projected Phase 2 hours;
- learning-rate selection with grid-edge flags;
- gates G1–G4 with their evidence;
- the Post-LN go/no-go decision and the Phase 2 recommendation.

**Local check (CPU):** the validator ran on the IWSLT14 config and every critical check passed: data counts, tokenizer fingerprint, exact parameter counts (36,663,296 / 36,565,136 / 36,606,112) and initial cross-entropy within its band. **Risk:** the CPU memory *estimate* for the worst-case batch (8,192 tokens of length 256) is 16,305 MiB for GRU + Bahdanau, above a T4's ~15,000 MiB; additive attention keeps a [B, T, S, 1104] tensor and grows with length squared. The Transformer (5,784) and BiLSTM + Luong (2,691) fit. The Kaggle validation step measures the real value before the pilot.

**Results:** none yet. Phase 1 runs on Kaggle (owner), then results go here.

---

## 2026-11-19 · Week 9 Phase 1 blocker: Kaggle preflight collection failure (fixed)

**Symptom:** `kaggle_validate.ipynb` stopped at the preflight tests. pytest could not collect the parity and numerical suites: `ModuleNotFoundError: No module named 'tests.parity'` (and `'tests.conftest'`).

**Root cause:**
- `tests/` had no `__init__.py`, so `tests` was an implicit namespace package.
- When Python resolves an import, a *regular* package (one with `__init__.py`) anywhere on `sys.path` beats namespace portions found earlier.
- The Kaggle image evidently contains some installed package that ships a top-level `tests/__init__.py`. It captured `import tests`, and `tests.parity` / `tests.conftest` do not exist in it.
- **Why it passed locally:** the local environment has no such package.
- **Reproduced locally** by putting a fake regular `tests` package on `PYTHONPATH`: identical errors.

**Fix:**
- `__init__.py` in `tests/` and every test subfolder. pytest's `pythonpath` puts the repository root first, so this repository's regular package wins.
- A policy test (`test_test_helpers_resolve_to_this_repository`) fails with a clear message if `tests` is ever shadowed again.
- `tests/unit/test_pilot.py` loads `scripts/pilot_report.py` by file path. `scripts` is an equally generic top-level name.
- **Verified:** with the fake shadowing package present, the policy, parity and numerical suites pass. Full suite: 641 passed.

**Lesson:** never rely on generic top-level names (`tests`, `scripts`, `utils`) as implicit namespace packages in environments with large, unknown site-packages.

---

## 2026-11-19 · Week 9 Phase 1: T4 validation and the GRU memory fix

**Kaggle validation (owner, T4, IWSLT14 config, worst-case batch at 8,192 tokens per side):**

| Model | Memory | Peak allocated | Target tok/s |
|---|---|---|---|
| transformer_iwslt | PASS | 4,723 MiB | 24,963 |
| bilstm_luong_iwslt | PASS | 1,990 MiB | 49,820 |
| gru_bahdanau_iwslt | **FAIL: CUDA out of memory** (allocation reached 10.24 GiB) | — | — |

- **The CPU estimate predicted this** (16.3 GB, entry above): additive attention keeps a `[B, T, S, 1104]` activation, which grows with length squared.
- **BiLSTM + Luong runs at twice the Transformer's tokens/s.** As on Multi30k, cuDNN recurrences are fast at this scale.
- **The Transformer's throughput matches Week 5's measurement** (24,669 tok/s).

**Fix (ADR-023 Amendment 1):** GRU + Bahdanau uses 4,096-token micro-batches × 2 accumulation steps. The effective update size (8,192 tokens), per-update token normalization and learning-rate schedule are identical to the other models. Fairness checks now compare data configs without the micro-batch size and verify the effective update size separately; tested in both directions (equal passes, smaller fails).

**Next:** re-validate GRU at 4,096 on the T4. If it passes, run the pilot. If it fails, use the 2,048 × 4 fallback.

---

## 2026-11-20 · Week 9 Phase 1: pilot stopped after its first run (fixed)

**Symptom (Kaggle):**
- `iwslt14_pilot_transformer_s1` trained to completion (6,000 updates, best validation NLL 3.526).
- No other run started. `STUDY.md` listed 1 completed run and `pilot report exit: 3`, with hours of session left.

**Root cause:** a crash in **incremental decoding under autocast** (`aiayn/inference/decoding.py`), not in `study.plan`, `study.runner` or `pilot.py`.
- **The crash:** `init_state` was called outside the autocast context. It projects the Transformer's cached cross-attention keys and values, so on the T4 they came out fp32, while each decoder step under fp16 autocast produced fp16 queries. The attention raised a dtype mismatch at the first evaluation of the first finished run.
- **How it surfaced:** the runner evaluates in-process, so the exception ended `run_study.py` with a traceback (non-zero exit).
- **Why it looked like a clean stop:** the notebook printed the exit code but continued into the report and publish cells.

**Why tests missed it:**
- every decoding test ran on CPU with autocast off;
- the fake-training study tests never decode.

**Reproduced locally** with CPU bf16 autocast: prefix decoding OK, incremental decoding `TypeError: dtype mismatch`. Simulating the real `week9_pilot` plan with fake training ran all 10 runs, which rules out the planner and runner.

**Fix:**
- `_init_state` builds the state under the same autocast as the encoder and decoder steps.
- New `tests/numerical/test_decoding_autocast.py` decodes under fp16 on CUDA (bf16 on CPU) for every model. It is part of the Kaggle preflight. Verified: it fails without the fix (4 Transformer cases) and passes with it.
- Study notebook: a crashed runner prints a banner, and the last cell fails after finished runs are reported and published.

**Impact:**
- **No retraining needed:** the completed Transformer pilot run is valid, since the bug was in decoding only.
- **Earlier results are unaffected:** Week 6–8 results predate incremental decoding.
- **Every GPU evaluation since Phase 0 would have crashed,** including `kaggle_train.ipynb`.

---

## 2026-11-21 · Week 9: Post-LN rejected at full budget; Pre-LN is the headline Transformer

**Full-budget comparison** (owner's server, `iwslt14_transformer` recipe: 30k updates, warmup 4,000, 8,192 tokens; seed 1; validation only):

| Run | Best valid NLL | Valid BLEU |
|---|---|---|
| `iwslt14_transformer-lr0.0005_s1` (Post-LN) | 3.034 | 5.79 |
| `iwslt14_transformer-pre_ln_s1` (Pre-LN, 7e-4) | **1.618** | **34.33** |

**Reading:**
- Post-LN does not catch up: from 3.54 at 6k updates it only reaches 3.03 at 30k. The "just slower" hypothesis is rejected for this setup.
- Pre-LN at 34.33 validation BLEU (sacreBLEU, cased) is in a sensible range for IWSLT'14 DE→EN.

**Decision (ADR-023 Amendment 2):**
- Pre-LN is the headline Transformer.
- **Phase 1c** selects learning rates at the full budget for all three models.
- Then the 3-seed Phase 2 benchmark.
- The Post-LN discrepancy with published results is a Week 10 investigation item.

**Corrections to my earlier diagnosis:**
- The fp16 and instability hypotheses were not supported by the logs (2 skipped updates, maximum gradient norm 2.95).
- The 6k-update gate was biased against slow-starting architectures.
