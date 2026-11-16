# Transformer From Scratch: Reproducing *Attention Is All You Need*

A from-scratch PyTorch implementation and reproduction of the original Transformer (Vaswani et al., 2017). The goal is not just to train a model but to **demonstrate and verify** every component: hand-written modules, numerical parity with PyTorch's reference implementations, fair RNN baselines, paper-style ablations with multiple seeds, and honest metric reporting.

> **Status: Week 5 of 11 complete (training system).** Label-smoothed loss, Noam/inverse-sqrt schedules, fp16 AMP with GradScaler, gradient accumulation, atomic checkpoints and **bit-exact resumption** across Kaggle sessions, plus JSONL/TensorBoard/W&B logging. Validated on a Kaggle Tesla T4: the IWSLT-size model trains at ≈ 25k target tokens/s within 2.2 GB of VRAM ([docs/KAGGLE.md](docs/KAGGLE.md)).

## What makes this different from a tutorial
| | |
|---|---|
| **Hand-written, policy-enforced** | `nn.Transformer`, `nn.MultiheadAttention`, `nn.LayerNorm` and SDPA are banned from `src/` by a CI test ([ADR-001](docs/decisions/ADR-001-manual-implementation.md)) |
| **Verified, not just trained** | Every component is parity-tested against PyTorch's reference implementations, forward and backward ([ADR-004](docs/decisions/ADR-004-parity-testing.md)) |
| **Faithful** | Post-LN, as in the paper, with Pre-LN as an ablation ([ADR-003](docs/decisions/ADR-003-post-ln-default.md)) |
| **Honest metrics** | sacreBLEU with signatures plus protocol-matched comparisons to published numbers ([ADR-002](docs/decisions/ADR-002-iwslt14-before-wmt14.md)) |

## Documentation
| Document | Contents |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Approved engineering specification (v1.1) |
| [docs/paper_notes.md](docs/paper_notes.md) | Section-by-section analysis, verified tables, claims register, underspecification register, 2017 vs modern |
| [docs/math.md](docs/math.md) | Derivations: attention, √d_k, multi-head, positional encoding, LayerNorm, Noam, label smoothing, parameter counts, complexity |
| [docs/decisions/](docs/decisions/README.md) | Architecture Decision Records |
| [docs/REPOSITORY.md](docs/REPOSITORY.md) | Folder contracts, ownership, import layering |
| [docs/experiment_log.md](docs/experiment_log.md) | Chronological lab notebook |
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
| 6 | Inference + evaluation |
| 7 | RNN baselines |
| 8 | IWSLT14 main benchmark (3 seeds, Kaggle) |
| 9 | Ablations |
| 10 | Analysis and visualization |
| 11 | Documentation and polish |

## Data quickstart
```bash
python scripts/inspect_data.py   --config configs/data/copy_task.yaml        # synthetic, no download
python scripts/prepare_data.py   --config configs/data/multi30k.yaml         # download + verify SHA-256
python scripts/train_tokenizer.py --config configs/data/multi30k.yaml        # joint BPE, 8k
python scripts/inspect_data.py   --config configs/data/multi30k.yaml --split valid
python scripts/count_params.py   --config configs/model/transformer_base.yaml --vocab-size 37000
python scripts/train.py --experiment configs/experiment/copy_task_small.yaml --run-dir runs/copy   # ~100% in minutes on CPU
python scripts/train.py --resume runs/copy --override train.max_updates=2000                     # exact continuation
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
