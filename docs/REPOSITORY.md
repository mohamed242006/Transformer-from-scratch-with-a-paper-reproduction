# Repository Architecture

This document is the **contract** for where code, data, configuration and documentation live. If a change does not fit this structure, update this document in the same change, or write an ADR for a significant restructuring.

## 1. Ownership model

This is currently a single-maintainer project. Ownership is assigned to **roles**, so that responsibilities stay clear and can be handed to collaborators later without restructuring. All roles are currently held by the **lead engineer**.

| Role | Owns | Accountable for |
|---|---|---|
| **Model owner** | `src/aiayn/modules/`, `src/aiayn/models/` | Architectural correctness, parity tests, faithfulness to the paper |
| **Data owner** | `src/aiayn/data/`, `configs/data/`, `data/` | Reproducible datasets, tokenization, no leakage, licences |
| **Training owner** | `src/aiayn/training/`, `configs/train/` | Optimization recipe, checkpointing, stability |
| **Evaluation owner** | `src/aiayn/evaluation/`, `src/aiayn/inference/` | Metric protocols, decoding, honest comparisons |
| **Infra owner** | `.github/`, `pyproject.toml`, `tests/policy/`, tooling configs, `src/aiayn/tracking/`, `src/aiayn/utils/` | CI, packaging, reproducibility tooling |
| **Research/docs owner** | `docs/`, `experiments/`, `notebooks/`, `README.md` | Narrative, ADRs, results reporting |

## 2. Folder tree

Status markers: **[W1]** created in Week 1 · *(Wn)* populated in that week.

```
.                                   # repository root
├── .github/
│   └── workflows/
│       └── ci.yml                  [W1] lint + policy/unit tests on CPU
├── .gitattributes                  [W1] line-ending normalization (Windows ↔ Linux)
├── .gitignore                      [W1]
├── .pre-commit-config.yaml         [W1]
├── pyproject.toml                  [W1] packaging, dependencies, ruff/pytest/mypy config
├── README.md                       [W1]
│
├── configs/                        [W1] (W2+) YAML configs, the only source of hyperparameters
│   ├── model/                      [W4] transformer_tiny.yaml, transformer_iwslt.yaml, transformer_base.yaml
│   ├── data/                       [W2] copy_task.yaml, multi30k.yaml · (W7) iwslt14_deen.yaml
│   ├── train/                      [W5] debug_cpu, copy_task, multi30k_t4, iwslt_t4, paper_base
│   └── experiment/                 [W5] copy_task_small, multi30k_tiny, debug_multi30k_cpu (ADR-018)
│
├── data/                           [W1] (W2+) local datasets and tokenizers, GITIGNORED except README
│   ├── raw/                        downloaded archives (checksummed)
│   ├── processed/                  cleaned, split, filtered text
│   └── tokenizers/                 SentencePiece .model/.vocab files
│
├── docs/                           [W1]
│   ├── DESIGN.md                   [W2] approved specification v1.1
│   ├── REPOSITORY.md               [W1] this file
│   ├── paper_notes.md              [W1]
│   ├── math.md                     [W1]
│   ├── experiment_log.md           [W1] chronological lab notebook
│   ├── KAGGLE.md                   [W5p] Kaggle setup, packaging, validation, expected VRAM
│   ├── kaggle_validation_notebook.md [W4] validation check design (implemented W5p)
│   ├── decisions/                  [W1] ADRs (index in decisions/README.md)
│   ├── figures/                    [W1] (W9+) final figures referenced by docs/README
│   └── results.md                  (W10) final results report
│
├── experiments/                    [W1] (W5+) one folder per run: metadata + metrics, no weights in git
│
├── notebooks/                      [W5] kaggle_validate.ipynb, kaggle_train.ipynb · (later) analysis
│
├── scripts/                        [W2–W5] prepare_data, train_tokenizer, inspect_data, count_params,
│                                   kaggle_validate, package_for_kaggle, train, evaluate,
│                                   translate, average_checkpoints, report, kaggle_publish_runs,
│                                   compare_runs
├── dist/                           (gitignored) build output: dist/kaggle/ dataset bundles (ADR-016)
│
├── src/
│   └── aiayn/                      [W1] installable package ("Attention Is All You Need")
│       ├── __init__.py             [W1] version only
│       ├── utils/                  [W2/W4] config.py, paths.py, hashing.py, seed.py, params.py
│       ├── modules/                [W3/W4] embedding, positional, attention, multihead, layernorm,
│       │                                feedforward, sublayer, masking, init, encoder, decoder
│       ├── models/                 [W4] config.py (ModelConfig, parameter formula), transformer.py
│       │   ├── base.py, factory.py [W7] EncoderDecoder interface; build_model dispatch
│       │   └── baselines/          [W7] RNNConfig, RNNSeq2Seq (LSTM/GRU), Bahdanau/Luong attention
│       ├── data/                   [W2] special_tokens, example (framing), config (schema), tokenizer,
│       │                                copy_task, datasets, sampler, collate, prepare, loading
│       ├── training/               [W5] config, experiment, losses, schedule, checkpoint, trainer
│       ├── inference/              [W6] decoding (greedy, beam), averaging, translator, evaluate, report;
│       │                           [W7] comparison (fairness checks, COMPARISON.md)
│       ├── evaluation/             [W6] metrics: sacreBLEU, lowercased BLEU, chrF, exact match, by-length;
│       │                           [W7] paired bootstrap significance
│       ├── tracking/               [W5] loggers.py: stdout, jsonl (source of truth), TensorBoard, W&B
│       └── diagnostics/            [W5p/W6] Kaggle validation, packaging, kaggle_runs (persistence)
│
└── tests/                          [W1]
    ├── conftest.py                 [W2] synthetic corpora, tiny tokenizer config, CharTokenizer
    ├── policy/                     [W1/W2] forbidden APIs (ADR-001), layering + sentencepiece rule
    ├── unit/                       [W2–W4] data, one file per module, encoder, decoder, transformer, model_config
    │                                    sampler, collate, prepare, seed
    ├── parity/                     [W3/W4] modules, layers + stacks, full model vs PyTorch (ADR-004)
    ├── numerical/                  [W3/W4] gradcheck (modules + full model), low-precision stability
    └── integration/                [W2/W4] data pipeline; data→model flow; save/load consistency
```

## 3. Package layering (import rules)

Dependencies point **downwards only**. Lower layers never import higher ones.

```
            scripts/   tests/            (may import anything public)
                 │
     ┌───────────┴────────────┐
 training/                inference/      (orchestration)
     │  ╲                    │
     │   ╲─ evaluation/      │            (evaluation works on strings and tensors, never on models)
     │   ╲─ tracking/        │
     │   ╲─ data/ ───────────┤
     ▼                       ▼
                models/                   (assembles modules)
                   │
                modules/                  (pure nn.Module building blocks)
                   │
                 utils/                   (no aiayn imports)
```

| Package | May import | Must **not** import |
|---|---|---|
| `utils` | stdlib, torch, numpy, yaml | any `aiayn.*` |
| `modules` | `utils`, torch | `models`, `data`, `training`, `inference`, `evaluation`, `tracking` |
| `models` | `modules`, `utils` | `data`, `training`, `inference`, `evaluation`, `tracking` |
| `data` | `utils`, sentencepiece, datasets | `modules`, `models`, `training`, `inference` |
| `evaluation` | `utils`, sacrebleu | `modules`, `models`, `training` |
| `tracking` | `utils`, wandb/tensorboard (optional) | `modules`, `models`, `data`, `training` |
| `inference` | `models`, `modules`, `data`, `evaluation`, `utils` | `training` (ADR-019: the evaluation pipeline decodes, then scores) |
| `training` | `models`, `data`, `evaluation`, `tracking`, `inference`, `utils` | none of the above restrictions |
| `diagnostics` | everything (platform validation and packaging sit on top) | — (nothing may import `diagnostics`) |

Enforcement: an AST-based layering test in `tests/policy/` (added in Week 2, when the first cross-package import exists).

## 4. Directory contracts

Each entry lists **Owner**, **Responsibilities**, **Acceptable contents** and **Forbidden contents**.

### `/` (repository root)
- **Owner:** Infra owner
- **Responsibilities:** project entry point; packaging and tooling configuration.
- **Acceptable:** `README.md`, `pyproject.toml`, `LICENSE`, tool configuration files (`.gitignore`, `.gitattributes`, `.pre-commit-config.yaml`), `CONTRIBUTING.md`.
- **Forbidden:** source code, notebooks, data, checkpoints, ad-hoc scripts, scratch files.

### `.github/workflows/`
- **Owner:** Infra owner
- **Responsibilities:** CI. Lint, type check, and the policy, unit, parity and integration test tiers on CPU.
- **Acceptable:** workflow YAML.
- **Forbidden:** secrets in plain text, GPU training jobs, steps that download full datasets.

### `configs/`
- **Owner:** shared. `model/` → Model owner, `data/` → Data owner, `train/` → Training owner, `experiment/` → Research owner.
- **Responsibilities:** the **single source of every hyperparameter**. A reported result must be reproducible from one `experiment/*.yaml` plus a git commit.
- **Acceptable:** YAML files, grouped by concern; composed experiment files that reference the others.
- **Forbidden:** Python code; secrets or API keys; absolute machine-specific paths; hyperparameters duplicated as literals in `src/` (a default may exist in a dataclass schema, but the YAML file is authoritative for runs).

### `data/` (gitignored)
- **Owner:** Data owner
- **Responsibilities:** local cache of raw downloads, processed corpora and trained tokenizers.
- **Acceptable:** files produced by `scripts/prepare_data.py` and `scripts/train_tokenizer.py`, checksum manifests, the `README.md`.
- **Forbidden in git:** everything except `README.md`. Dataset licences (for example TED / IWSLT: CC BY-NC-ND) do not permit redistribution, and the files are large.

### `docs/`
- **Owner:** Research/docs owner
- **Responsibilities:** the engineering narrative, covering *why*, not only *what*.
- **Acceptable:** Markdown documents, ADRs, final figures (`figures/`, PNG/SVG, < 1 MB each), the results report.
- **Forbidden:** code that must be executed to understand the docs; raw experiment dumps; screenshots of numbers that are not also in `experiments/`.

### `docs/decisions/`
- **Owner:** Lead engineer (decider of record)
- **Responsibilities:** immutable record of significant decisions.
- **Acceptable:** `ADR-NNN-kebab-title.md` files following the template; the index `README.md`.
- **Forbidden:** editing the body of an accepted ADR (supersede it instead); decisions without alternatives considered.

### `experiments/`
- **Owner:** Research/docs owner
- **Responsibilities:** provenance for every reported number.
- **Acceptable:** per-run folders `YYYY-MM-DD_<short-name>/` containing the frozen resolved config, `metadata.json` (git commit, dirty flag, seed, library versions, GPU, data and tokenizer hashes), `metrics.json`, a short `NOTES.md` (hypothesis → result → conclusion) and test-set hypothesis files (text, small).
- **Forbidden in git:** checkpoints/weights (`*.pt`, `*.safetensors`), TensorBoard event files, W&B local caches, files > 5 MB. These are gitignored and stored as W&B artifacts or locally.

### `notebooks/`
- **Owner:** Research/docs owner
- **Responsibilities:** explanation and analysis (derivations with toy tensors, attention visualizations, results figures).
- **Acceptable:** numbered notebooks (`01_…`) that **import from `aiayn`**; outputs cleared before commit, except final figure notebooks.
- **Forbidden:** defining model or training logic that `src/` lacks (notebooks are never the source of truth); long training runs; hard-coded results that are not loaded from `experiments/`.

### `scripts/`
- **Owner:** the owner of the package each script drives
- **Responsibilities:** thin command-line entry points: `prepare_data`, `train_tokenizer`, `train`, `evaluate`, `translate`, `average_checkpoints`, `benchmark`.
- **Acceptable:** argument parsing, config loading, calls into `aiayn`. Each script has a `main()` and an `if __name__ == "__main__":` guard (**required on Windows**, where DataLoader workers use the `spawn` start method).
- **Forbidden:** business logic (if it is more than ~30 lines of logic, it belongs in `src/`); hard-coded hyperparameters.

### `src/aiayn/`
- **Owner:** see the sub-packages
- **Responsibilities:** the installable library. **All importable logic lives here.**
- **Acceptable:** typed, documented Python modules organized by the layering rules.
- **Forbidden (package-wide):** the APIs banned by ADR-001 (`nn.Transformer*`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.LayerNorm`, `F.layer_norm`, Hugging Face `transformers`); file or network I/O at import time; global mutable state; `print` for logging (use `logging` or the tracker); notebook-style code.

### `src/aiayn/utils/`
- **Owner:** Infra owner
- **Responsibilities:** cross-cutting helpers with **no domain logic**: seeding and determinism, config loading and validation, parameter counting, timing and profiling, device selection.
- **Acceptable:** small, pure helpers.
- **Forbidden:** anything model-specific. **Masks belong in `modules/`**, because they define model semantics and are parity-tested. Must not become a dumping ground: a helper used by only one package lives in that package.

### `src/aiayn/modules/`
- **Owner:** Model owner
- **Responsibilities:** the hand-written building blocks: token embedding, positional encodings, scaled dot-product attention, multi-head attention, LayerNorm, FFN, the sublayer (residual + dropout + norm) connection, masking utilities, encoder and decoder layers.
- **Acceptable:** `nn.Module` subclasses and pure tensor functions with documented tensor shapes (`[B, T, d]`).
- **Forbidden:** data loading, tokenization, loss functions, optimizers, logging side effects, config-file parsing (modules take explicit constructor arguments).

### `src/aiayn/models/`
- **Owner:** Model owner
- **Responsibilities:** assembling modules into complete models: `Transformer` (encode / decode / generator, weight tying, initialization) and `baselines/` (LSTM, GRU+Bahdanau, BiLSTM+Luong; ADR-021). Every model implements `EncoderDecoder` (`base.py`); `build_model` (`factory.py`) dispatches on the config, and `nn.LSTM`/`nn.GRU` may appear only in `baselines/`.
- **Acceptable:** model classes and their factory functions built from a config dataclass.
- **Forbidden:** training loops, decoding algorithms (those live in `inference/`), dataset code.

### `src/aiayn/data/`
- **Owner:** Data owner
- **Responsibilities:** tokenizer wrapper (the **only** place `sentencepiece` is imported, ADR-005), dataset readers, length filtering, the token-budget bucketed batch sampler, collation and padding, the teacher-forcing shift.
- **Acceptable:** `Dataset`/`Sampler` classes, picklable (Windows spawn), deterministic given a seed.
- **Forbidden:** model code; tokenizer training on non-training splits; network downloads at import time.

### `src/aiayn/training/`
- **Owner:** Training owner
- **Responsibilities:** the training loop (gradient accumulation, AMP, NaN guard, periodic evaluation), the label-smoothed loss, the Noam scheduler, checkpoint save/load/rotation.
- **Acceptable:** orchestration plus objective and schedule implementations (hand-written, ADR-005 §5).
- **Forbidden:** model architecture definitions; metric implementations (call `evaluation/`); direct W&B calls (use `tracking/`).

### `src/aiayn/inference/`
- **Owner:** Evaluation owner
- **Responsibilities:** greedy decoding, beam search with GNMT length penalty, checkpoint averaging, the `translate()` API, and (extension) the KV cache.
- **Acceptable:** decoding algorithms operating on a model's `encode`/`decode` interface.
- **Forbidden:** training-time code; metric computation.

### `src/aiayn/evaluation/`
- **Owner:** Evaluation owner
- **Responsibilities:** sacreBLEU/chrF wrappers that record signatures, perplexity aggregation, length-bucketed analysis, paired bootstrap significance tests.
- **Acceptable:** functions over **strings and numbers**, independent of any model class.
- **Forbidden:** importing models; home-made BLEU used for *reported* numbers (an educational implementation may exist only if tested against sacreBLEU).

### `src/aiayn/tracking/`
- **Owner:** Infra owner
- **Responsibilities:** a `Logger` protocol with stdout, TensorBoard and W&B backends; run metadata capture (git hash, environment).
- **Acceptable:** thin adapters; optional dependencies imported lazily.
- **Forbidden:** metric computation; hard dependencies on W&B (the project must run fully offline).

### `src/aiayn/diagnostics/`
- **Owner:** Infra owner
- **Responsibilities:** platform validation for Kaggle and local machines (environment, dependencies, CUDA, data, tokenizer, model, VRAM, throughput) and Kaggle Dataset packaging (ADR-016).
- **Acceptable:** check functions returning `Outcome`, report serialization, packaging helpers.
- **Forbidden:** training logic (it belongs in `training/`); being imported by any other package; network access other than the optional reachability probe and pinned downloads.

### `tests/`
- **Owner:** shared. Each test directory mirrors the owner of the code under test. `policy/` belongs to the Infra owner.
- **Responsibilities:** evidence of correctness that does not depend on training outcomes.
- **Acceptable:** pytest files `test_*.py`; fixtures in `conftest.py`; reference implementations from PyTorch (ADR-004); tiny synthetic data generated in-test.
- **Forbidden:** network access; dependence on `data/` contents (tests generate their own data); GPU-only tests without the `gpu` marker; tests slower than ~10 s without the `slow` marker; loosening ADR-004 tolerances.

| Sub-directory | Marker | Runs in CI | Purpose |
|---|---|---|---|
| `policy/` | none | yes | Repository rules (forbidden APIs, layering) |
| `unit/` | none | yes | Behaviour, shapes, invariances, closed forms |
| `parity/` | `parity` | yes | Numerical equality with PyTorch references |
| `numerical/` | `numerical` | yes | gradcheck, low-precision safety |
| `integration/` | `integration`, `slow` | smoke subset | Overfit, copy task, end-to-end pipeline |

## 5. Naming conventions
- Python modules: `snake_case.py`. Classes: `PascalCase`. Constants: `UPPER_SNAKE`.
- Configs: `<thing>_<variant>.yaml` (e.g. `transformer_iwslt.yaml`).
- Experiments: `YYYY-MM-DD_<model>-<data>-<variant>` (e.g. `2026-12-20_transformer-iwslt14-postln-s1`).
- ADRs: `ADR-NNN-kebab-case-title.md`.
- Tensor-shape docstrings use the symbols in [math.md §0](math.md#0-notation-and-conventions): `B, S, T, d, h, d_k, V`.
