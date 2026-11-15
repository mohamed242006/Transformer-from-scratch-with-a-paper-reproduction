# configs/

The single source of every hyperparameter. A reported result must be reproducible from **one** `experiment/*.yaml` plus a git commit.

| Sub-folder | Owner | Contents |
|---|---|---|
| `model/` | Model owner | Architecture: N, d_model, d_ff, h, dropout, norm_position, PE type |
| `data/` | Data owner | Dataset source + SHA-256 checksums, filtering, tokenizer settings (ADR-005) |
| `train/` | Training owner | Optimizer, schedule, token budget, accumulation, AMP, checkpoint cadence |
| `experiment/` | Research owner | Composed configs: one file per reported result or ablation |

Rules: YAML only; no secrets; no absolute paths. The config system itself is decided in ADR-007 (Week 2).
Full contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#configs).
