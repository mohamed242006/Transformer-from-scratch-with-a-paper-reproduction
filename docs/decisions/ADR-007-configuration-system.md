# ADR-007: Configuration system: frozen dataclasses + YAML with strict validation

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [REPOSITORY.md `configs/`](../REPOSITORY.md#configs), spec §9

## Context
`configs/` is the single source of hyperparameters (REPOSITORY.md). We need typed, validated, reproducible configuration with command-line overrides, usable from scripts, tests and notebooks, with Python 3.10–3.12 support and minimal dependencies. The main failure mode to prevent is a **silently ignored typo** (for example `max_token: 2048`, which leaves the default in place and produces an unexplained result).

## Decision
1. **Schemas are frozen `dataclasses`** owned by the package that consumes them (for example `aiayn.data.config.DataConfig`). Validation that spans fields lives in `__post_init__`.
2. **Files are YAML**, parsed with `yaml.safe_load`, and converted by a generic, strict converter `aiayn.utils.config.from_dict`:
   - **unknown keys → error** (with the dotted path and the allowed keys);
   - missing required keys → error;
   - types checked: `bool` is never accepted as `int`, `int` is accepted as `float`, `Literal` values are enforced, `X | None`, nested dataclasses, `tuple[X, ...]`, `list[X]`, `dict[str, X]`.
3. **Overrides** use dotted `key=value` strings, with each value parsed as YAML (`batching.max_tokens=2048`, `filter.max_len_ratio=null`). An override must target an existing parent mapping.
4. **Relative paths** in configs resolve against the **repository root** (located by searching upwards for `pyproject.toml`), never against the current directory.
5. **Freezing:** `config_to_dict` serializes the resolved configuration, and every run stores it (spec §11).
6. **Composition** across groups (`data/`, `model/`, `train/`) uses an experiment file that names one file per group. It is implemented with the first schema that needs it (Week 5). Week 2 only needs the `data` group.

## Consequences
**Positive:** typos fail fast; mypy-typed access (`cfg.batching.max_tokens`); zero framework dependency; trivial to test; schemas sit next to their code. **Negative:** no built-in sweeps or interpolation (Hydra has these) → sweeps will be driven by W&B or simple scripts. The converter is about 150 lines we own and must test. **Risk:** schema and YAML drift → a test loads **every** file in `configs/` against its schema.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Hydra / OmegaConf | Powerful (composition, sweeps) but heavy, magic working-directory behaviour, weaker static typing; overkill for this project's scale |
| argparse only | Hyperparameters scattered across CLIs; no files to version |
| pydantic | Excellent validation, but an extra dependency, and its v1/v2 differences add churn; dataclasses suffice |
| Plain dicts | No typing, typos pass silently |

## Compliance
`tests/unit/test_config.py` covers the converter; `test_repository_configs_are_valid` loads every `configs/data/*.yaml`.
