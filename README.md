# Transformer From Scratch: Reproducing *Attention Is All You Need*

A from-scratch PyTorch implementation and reproduction of the original Transformer (Vaswani et al., 2017). The goal is not just to train a model but to **demonstrate and verify** every component: hand-written modules, numerical parity with PyTorch's reference implementations, fair RNN baselines, paper-style ablations with multiple seeds, and honest metric reporting.

> **Status: v1.0, complete.** A from-scratch Transformer, verified against PyTorch references, beats parameter-matched attention RNNs on **IWSLT'14 DE→EN (33.10 vs 32.04 BLEU, 3 seeds)** and on Multi30k EN→DE. The comparison protocol (fairness rules and statistics) was fixed before the runs. Full results: [docs/results.md](docs/results.md).

## What makes this different from a tutorial
| | |
|---|---|
| **Hand-written, policy-enforced** | `nn.Transformer`, `nn.MultiheadAttention`, `nn.LayerNorm` and SDPA are banned from `src/` by a CI test ([ADR-001](docs/decisions/ADR-001-manual-implementation.md)) |
| **Verified, not just trained** | Every component is parity-tested against PyTorch's reference implementations, forward and backward ([ADR-004](docs/decisions/ADR-004-parity-testing.md)) |
| **Faithful, with declared deviations** | Post-LN as in the paper is the default ([ADR-003](docs/decisions/ADR-003-post-ln-default.md)) and is used for all Multi30k results. The IWSLT'14 headline model is **Pre-LN**, because 6+6 Post-LN did not train in this recipe. The failure is bounded, not explained ([ADR-023 Amendment 4](docs/decisions/ADR-023-iwslt14-benchmark.md)) |
| **Honest metrics** | sacreBLEU with signatures plus protocol-matched comparisons to published numbers ([ADR-002](docs/decisions/ADR-002-iwslt14-before-wmt14.md)) |

## Results

**IWSLT'14 DE→EN** (headline). Test set: 6,750 sentences. Parameters are matched within 0.3% (≈ 36.6M). Results are mean ± std over 3 seeds, sacreBLEU (cased).

| Model | Test BLEU, validation-selected | Test BLEU, protocol defaults | tok-lc BLEU |
|---|---|---|---|
| **Transformer** (6+6, d = 512, Pre-LN) | **33.10 ± 0.07** | **32.95 ± 0.05** | 34.84 |
| BiLSTM + Luong attention | 32.04 ± 0.10 | 31.80 ± 0.19 | 33.83 |
| BiGRU + Bahdanau attention | 28.88 ± 0.25 | 11.51 ± 14.37 † | 30.78 |

**Multi30k EN→DE** (small-scale study with ablations; 7.6M parameters, 5 seeds, protocol defaults): Transformer **36.71 ± 0.20**, BiGRU + Bahdanau 33.97 ± 0.15, BiLSTM + Luong 32.46 ± 0.51, LSTM without attention 12.60 ± 0.26.

- **The Transformer wins on both benchmarks.** Every gap is significant: two-level paired bootstrap over sentences and seeds, Holm-corrected.
- **The margin depends on tuning.** With a learning rate selected for each model, the lead over the best RNN is **≈ 1 BLEU** on IWSLT'14 (Δ −1.06, 95% CI [−1.33, −0.82]). Under shared untuned defaults on Multi30k it was 2.7–4.3.
- **Attention is the decisive ingredient:** removing it costs 19–24 BLEU on Multi30k.
- **Normalization placement depends on depth.** Post-LN beats Pre-LN at 3+3 layers (+1.1 BLEU, Multi30k). At 6+6 on IWSLT'14 it reached only 5.79 validation BLEU vs Pre-LN's 34.33.
- **The study's safeguards caught failures that a single number would hide:** a silently collapsed BiLSTM seed (Multi30k) and the GRU's late-training divergence on IWSLT'14.
- **Literature scale:** tok-lc BLEU of 34.67–34.84 puts the Transformer in the published range for 6+6 IWSLT'14 models (≈ 34–35). This is comparable to those results, not identical, because our models train on raw cased text.

† GRU + Bahdanau diverged late in training in 2 of 3 seeds, so its averaged checkpoint collapsed (0.06 / 27.63 / 6.84). The validation-selected best checkpoint is sound. Validation-selected means each model's checkpoint and beam/α were chosen on validation, never on test. Protocol defaults are the averaged checkpoint, beam 4, α 0.6.

Evidence for every IWSLT'14 number (configs, training logs, evaluations, test hypotheses) is in [experiments/2026-11-22_iwslt14-week9-benchmark/](experiments/2026-11-22_iwslt14-week9-benchmark/NOTES.md). Training all 24 IWSLT'14 runs took 8.9 GPU-hours on one RTX 5070 Ti.

## Documentation
| Document | Contents |
|---|---|
| [docs/results.md](docs/results.md) | **Final results:** IWSLT'14 benchmark, Multi30k study and ablations, statistics, limitations, conclusions |
| [docs/DESIGN.md](docs/DESIGN.md) | Approved engineering specification (v1.1) |
| [docs/paper_notes.md](docs/paper_notes.md) | Section-by-section analysis, verified tables, claims register, underspecification register, 2017 vs modern |
| [docs/math.md](docs/math.md) | Derivations: attention, √d_k, multi-head, positional encoding, LayerNorm, Noam, label smoothing, parameter counts, complexity |
| [docs/decisions/](docs/decisions/README.md) | Architecture Decision Records |
| [docs/REPOSITORY.md](docs/REPOSITORY.md) | Folder contracts, ownership, import layering |
| [docs/experiment_log.md](docs/experiment_log.md) | Chronological lab notebook |
| [experiments/](experiments/README.md) | Archived evidence behind reported numbers |
| [docs/KAGGLE.md](docs/KAGGLE.md) | Running on Kaggle: setup, dataset packaging, validation notebook, expected VRAM |
| [docs/kaggle_validation_notebook.md](docs/kaggle_validation_notebook.md) | Design of the Kaggle validation checks |

## Plan
| Week | Milestone |
|---|---|
| 1 ✅ | Paper analysis, derivations, ADRs, repository architecture |
| 2 ✅ | Data pipeline (copy task, Multi30k, SentencePiece, token-budget batching) |
| 3 ✅ | Core modules + parity tests |
| 4 ✅ | Full model + invariance tests |
| 5 ✅ | Training system |
| 6 ✅ | Inference + evaluation |
| 7 ✅ | RNN baselines and fairness protocol (Multi30k) |
| 8 ✅ | Multi-seed study, decoding and training sensitivity, ablations (Multi30k) |
| 9 ✅ | IWSLT'14 DE→EN benchmark: pinned data, pilot gate, full-budget LR selection, 3 seeds × 3 models |
| 10 ◐ | Analysis: Post-LN result bounded (ADR-023 Amendment 4); figure set deferred to future work |
| 11 ✅ | Documentation and v1.0 release |

## Future work
- **Explain the 6+6 Post-LN failure:** per-layer gradient norms at initialization, then one-factor runs over Adam ε, weight decay, Q/K/V init gain, tokens per update and weight tying.
- **GRU + Bahdanau late-training divergence:** a full-budget learning-rate check.
- **A wider decoding grid:** every model selected the corner of the grid (beam 8, α 1.0).
- **Figures:** training curves, attention maps, positional-encoding heatmaps, gradient norms.
- **Scale:** more seeds; WMT'14 EN→DE with the base model.

## Reproduce the IWSLT'14 benchmark
```bash
python scripts/prepare_data.py --config configs/data/iwslt14.yaml                   # download + pinned SHA-256 and split counts
python scripts/run_study.py --study configs/study/week9_benchmark.yaml --work runs  # trains and evaluates 3 models x 3 seeds (resumable)
python scripts/analyze_study.py --study configs/study/week9_benchmark.yaml --work runs --out runs/STUDY.md
```
The hyperparameters are frozen in `configs/experiment/iwslt14_{transformer,bilstm_luong,gru_bahdanau}.yaml`. Expect about 6 GPU-hours on a 16 GB consumer GPU. Running on Kaggle is covered in [docs/KAGGLE.md](docs/KAGGLE.md).

## Data quickstart
```bash
python scripts/inspect_data.py   --config configs/data/copy_task.yaml        # synthetic, no download
python scripts/prepare_data.py   --config configs/data/multi30k.yaml         # download + verify SHA-256
python scripts/train_tokenizer.py --config configs/data/multi30k.yaml        # joint BPE, 8k
python scripts/inspect_data.py   --config configs/data/multi30k.yaml --split valid
python scripts/count_params.py   --config configs/model/transformer_base.yaml --vocab-size 37000
python scripts/train.py --experiment configs/experiment/copy_task_small.yaml --run-dir runs/copy   # ~100% in minutes on CPU
python scripts/train.py --resume runs/copy --override train.max_updates=2000                     # exact continuation
python scripts/evaluate.py --run-dir runs/copy --greedy                                          # sacreBLEU, chrF, exact match
```

## Development setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest                 # policy, unit and integration tests
mypy                   # strict type check
ruff check . && ruff format --check .
```

Python ≥ 3.10. GPU training runs are intended for Linux or WSL2; development and tests work on Windows.

## Reference
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
