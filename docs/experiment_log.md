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
