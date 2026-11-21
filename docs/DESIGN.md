# Engineering & Research Specification

| | |
|---|---|
| **Project** | Transformer From Scratch: reproduction of *Attention Is All You Need* (Vaswani et al., 2017) |
| **Version** | **1.1** (v1.0 approved in conversation 2026-11-15; v1.1 merges the Week 1 amendments) |
| **Status** | Approved. Changes go through ADRs ([decisions/](decisions/README.md)). |
| **Companion docs** | [paper_notes.md](paper_notes.md) · [math.md](math.md) · [REPOSITORY.md](REPOSITORY.md) |

Other documents cite this file as "spec §N". Section numbers are stable.

### Changelog
| Version | Change |
|---|---|
| 1.0 | Initial specification (sections 1–14) |
| 1.1 | Week 1 amendments A1–A10 ([experiment_log.md](experiment_log.md)): embedding init N(0, d^-½); overfit tests with ε_ls = 0; initial-loss band; label smoothing ε/K; package `aiayn`; `data/`, `tests/parity/`, `tests/policy/` added; masks in `modules/`; dual BLEU reporting; step-based checkpoints. Week 2 pre-flight: ruff pinned, layering policy enforced. |

---

## 1. Project overview
**Why it matters.** The Transformer underlies nearly every modern AI system. This project shows understanding *from the tensors up*, and the ability to **reproduce a published result rigorously**, which is the core research-engineering skill.

**Skills demonstrated.** Deep-learning fundamentals (hand-written attention, normalization, masking, initialization) · mathematical maturity (derivations in math.md) · research engineering (faithful reproduction, ablations, seeds, significance tests) · ML systems (token batching, mixed precision, accumulation, checkpoint averaging, beam search) · software engineering (typed package, configs, tests, CI) · communication (ADRs, logs, report).

**How it differs from using Hugging Face.** Nothing is loaded pre-trained or hidden: every layer is written by hand and parity-tested; training starts from random initialization, so optimization problems (warmup, Post-LN instability) are faced directly; masking, shifting and decoding are implemented, not configured; evaluation protocols are designed and justified.

## 2. Technical scope
### 2.1 Implemented manually
Token embeddings (√d scaling, three-way tying) · sinusoidal PE (+ learned variant) · scaled dot-product attention · multi-head attention · residual + dropout + LayerNorm wrapper (Post-/Pre-LN) · hand-written LayerNorm · position-wise FFN · encoder and decoder layers/stacks · full Transformer · masks · label-smoothed loss · Noam schedule · training loop (token batching, accumulation, AMP, checkpointing) · checkpoint averaging · greedy and beam search (GNMT length penalty) · evaluation pipeline · RNN baselines with hand-written Bahdanau/Luong attention.

### 2.2 External libraries
Policy: anything that **is** the Transformer is hand-written. Standard infrastructure, or tools whose reimplementation would make results incomparable, use libraries. See [ADR-001](decisions/ADR-001-manual-implementation.md) and [ADR-005](decisions/ADR-005-sentencepiece-tokenization.md).

| Library | Use |
|---|---|
| `torch` core (`nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.Dropout`, `optim.Adam`, autograd) | Allowed |
| `nn.Transformer*`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.LayerNorm`, HF `transformers` | **Forbidden in `src/`**, allowed in `tests/` as references (enforced by a policy test) |
| `nn.LSTM` / `nn.GRU` | Baselines only |
| `sentencepiece` | Tokenization, imported only in `aiayn.data` |
| `sacrebleu` | Required for reported BLEU |
| `pyyaml` + dataclasses | Configuration ([ADR-007](decisions/ADR-007-configuration-system.md)) |
| `wandb`, `tensorboard`, `matplotlib`, `pytest`, `hypothesis`, `ruff`, `mypy` | Tooling |

## 3. Paper analysis
Fully covered in [paper_notes.md](paper_notes.md) (section-by-section analysis, Tables 1–4 checked against arXiv v7, a claims register, an underspecification register) and [math.md](math.md) (derivations).

**Reproduced exactly:** Post-LN architecture · scaled dot-product and multi-head attention · sinusoidal PE · embedding scaling + tying with a shared vocabulary · Adam (0.9, 0.98, 1e-9) · Noam schedule · label smoothing 0.1 · residual and embedding dropout · beam 4, α = 0.6, max length = input + 50 · checkpoint averaging · token-budget batching · selected Table 3 ablations.
**Simplified:** dataset (IWSLT14/Multi30k instead of WMT14, [ADR-002](decisions/ADR-002-iwslt14-before-wmt14.md)) · model size for small data · batch via accumulation · steps/warmup scaled · one GPU · big model, EN-FR and parsing skipped.

## 4. Architecture breakdown
Per-module purpose, shapes and pitfalls: [math.md](math.md) §1–§6 and §9; [paper_notes.md](paper_notes.md) §4. Shape symbols: `B, S, T, d, h, d_k, V` ([math.md §0](math.md#0-notation-and-conventions)).

| Module | Input → output |
|---|---|
| Embedding | `[B, L]` long → `[B, L, d]` (× √d) |
| Positional encoding | `[B, L, d]` → `[B, L, d]` (+ PE buffer, dropout) |
| Scaled dot-product attention | Q `[B,h,Tq,d_k]`, K/V `[B,h,Tk,d_k]`, mask `[B|1,1,Tq|1,Tk]` → `[B,h,Tq,d_k]` |
| Multi-head attention | `[B,Tq,d]`, `[B,Tk,d]` → `[B,Tq,d]` |
| Encoder layer | `[B,S,d]` + src mask `[B,1,1,S]` → `[B,S,d]` |
| Decoder layer | `[B,T,d]`, memory `[B,S,d]`, tgt mask `[B,1,T,T]`, src mask `[B,1,1,S]` → `[B,T,d]` |
| Transformer | src `[B,S]`, tgt_in `[B,T]` → logits `[B,T,V]` |

## 5. Dataset selection
Ladder: **synthetic copy/reverse** (correctness) → **Multi30k EN→DE** (development) → **IWSLT14 DE→EN** (primary benchmark) → WMT14 EN→DE (optional stretch). Optional side track: WikiText-2 decoder-only LM. Rationale and comparison: [ADR-002](decisions/ADR-002-iwslt14-before-wmt14.md). Tokenization: joint SentencePiece BPE trained on the training split only (~8k Multi30k, ~10k IWSLT, ~32–37k WMT).

## 6. Training strategy
### 6.1 Core recipe
Adam (0.9, 0.98, 1e-9) · Noam schedule (step = optimizer update, counted from 1) · label-smoothed CE (ε = 0.1, ε/K convention, pad positions excluded, per-token normalization) · token-budget bucketed batches · gradient accumulation · bf16/fp16 AMP · no gradient clipping by default (norm logged) · step-based checkpoints, average of the last 5 · model selection on validation only · ≥ 3 seeds for headline claims.

### 6.2 Configurations
| | Beginner | Mid-level | Portfolio |
|---|---|---|---|
| Data | Copy → Multi30k | IWSLT14 DE→EN | WMT14 EN→DE (or subset) |
| N / d / d_ff / h | 3 / 256 / 1024 / 4 | 6 / 512 / 1024 / 4 | 6 / 512 / 2048 / 8 |
| Dropout | 0.1 | 0.3 | 0.1 |
| Vocabulary | ~8k | ~10k | ~37k |
| Effective batch | ~4k tokens | ~4–16k tokens | ~25k + 25k tokens |
| LR | Noam shape, peak 5e-4–1e-3, warmup 5–10% | inverse-sqrt, peak 5e-4, warmup 4000 | exact Noam, warmup 4000 |
| Parameters ([math.md §9](math.md#9-parameter-count-formulas)) | 7.58M | 36.66M | 63.08M |

## 7. Baseline models
LSTM seq2seq (no attention) · GRU + Bahdanau attention · BiLSTM + Luong attention (strongest RNN) · Transformer ablations as internal baselines.
### 7.3 Fairness protocol
Same data, tokenizer and splits; parameters matched within ±10%; same token budget and update count (or wall-clock, reported); the same small LR sweep for every model; identical decoding and evaluation. **Metrics:** sacreBLEU (mean ± std, 3 seeds) + chrF, validation CE/perplexity (unsmoothed), BLEU by length bucket, tokens/s, time-to-BLEU, latency, memory, parameters.

## 8. Evaluation metrics
Primary: **sacreBLEU** (detokenized, cased, signature) plus BLEU under the reference protocol for literature comparisons ([ADR-006](decisions/ADR-006-evaluation-protocol.md)). Secondary: chrF. Diagnostics: validation CE, perplexity = exp(unsmoothed NLL) (comparable only under an identical tokenizer), teacher-forced token accuracy. Copy task: sequence exact match. Significance: paired bootstrap resampling.

## 9. Engineering structure
Authoritative: [REPOSITORY.md](REPOSITORY.md) (tree, directory contracts, ownership, import layering).

## 10. Testing strategy
Unit (shapes, invariances, closed forms) · parity against PyTorch references ([ADR-004](decisions/ADR-004-parity-testing.md)) · numerical (float64 gradcheck, fp16/bf16, NaN guards) · integration (overfit with ε_ls = 0, copy task 100%, save/load, decoding consistency, CPU end-to-end smoke) · policy (forbidden APIs, layering). CI runs everything not marked `slow`/`gpu`.

## 11. Experiment tracking
W&B (primary) and TensorBoard (offline) behind one `Logger` interface. Logged: run metadata (config, commit, seed, versions, data and tokenizer hashes); per-step loss, LR, gradient norm, tokens/s, memory; per-evaluation validation metrics and sample translations; per-head attention entropy; final test metrics.
### 11.3 Figures
Loss curves · LR schedule · BLEU by length bucket · throughput and memory vs length · attention heatmaps · PE heatmap and similarity · ablation bars · per-layer gradient norms Post-/Pre-LN · BLEU vs GPU-hours.

## 12. Resume value
CV bullets only with numbers that exist in `docs/results.md` with linked runs. Most persuasive evidence: parity tests, an honest comparison with published numbers, ablations with error bars, the Post-LN warmup experiment, length-bucket and complexity analyses, a clear README.

## 13. Roadmap
| Week | Milestone | Exit criterion |
|---|---|---|
| 1 ✅ | Paper, math, ADRs, repository | Derivations complete; policy test in CI |
| 2 ✅ | Data pipeline: copy task, Multi30k, SentencePiece, token-budget batching, collation, teacher forcing, masks, config | Copy task and Multi30k batches correct; masks verified; all tests and CI green |
| 3 ✅ | Core modules + parity tests | Module parity passes |
| 4 ✅ | Full model | Overfits one batch; full-stack parity |
| 5 ✅ | Training system | Copy task 100%; Multi30k converges |
| 6 ✅ | Inference + evaluation | First BLEU; beam=1 equals greedy |
| 7 ✅ | Baselines (ADR-021) | All baselines through the same pipeline |
| 8 ✅ | Multi30k study: multi-seed, sensitivity, ablations, statistics (ADR-022) | 5-seed comparison, Table-3-style ablations, significance, final experimental report |
| 9 | IWSLT14 benchmark (ADR-023): Phase 0 ✅ data, metrics, incremental decoding · Phase 1 ✅ pilot + full-budget Post-LN test → **Pre-LN headline** (Amendment 2) · Phase 1c full-budget learning-rate selection · Phase 2 3-seed benchmark vs GRU and BiLSTM | Headline table on IWSLT14 (sacreBLEU + tok-lc BLEU), significance |
| 10 | Analysis and figures, including the deep Post-LN investigation (ADR-023 Amendment 2, open item) | §11.3 figures; Post-LN result explained or bounded |
| 11 | Documentation | Reproducible from a fresh clone |
| 12+ | Extensions (WMT14 removed from scope, ADR-010) | |

## 14. Final review and extensions
Risks: a crowded genre (differentiate by rigor), a 2017 architecture (add the modernization ladder), small scale, compute, scope creep.
### 14.3 Extensions (ranked)
1. KV cache · 2. "2017 → modern" ablation ladder (Pre-LN, RMSNorm, SwiGLU, RoPE, GQA) · 3. attention efficiency study (SDPA/FlashAttention/Triton) · 4. mini scaling-law study · 5. DDP/FSDP · 6. hand-written BPE · 7. relative/rotary PE + length extrapolation · 8. head pruning and interpretability · 9. ONNX/INT8/serving/demo · 10. COMET and human evaluation · 11. technical blog post.
