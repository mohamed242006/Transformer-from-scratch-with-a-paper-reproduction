# scripts/

Thin CLI entry points: argument parsing + config loading + calls into `aiayn`. Any logic longer than ~30 lines belongs in `src/`. Run any script with `--help` for its options.

| Script | Purpose | Notebook counterpart |
|---|---|---|
| `prepare_data.py` | Download a dataset (pinned URL and SHA-256) and build its splits | `reproduce_results`, `kaggle_validate`; automatic via `train.py --prepare-data` |
| `train_tokenizer.py` | Train the joint SentencePiece tokenizer (deterministic; ADR-005) | `reproduce_results`, `kaggle_validate` |
| `inspect_data.py` | Print split statistics and sample batches | — |
| `count_params.py` | Parameter count of a model config | — |
| `train.py` | Train or resume one run (`--experiment` / `--resume`, `--override`) | `kaggle_train` |
| `evaluate.py` | Decode a split and score it (sacreBLEU, chrF, tok-lc BLEU, BLEU by length) | `kaggle_train`, `kaggle_study` |
| `translate.py` | Translate sentences with a trained run | — |
| `average_checkpoints.py` | Average a run's checkpoint snapshots | — (done automatically by `evaluate.py --checkpoint averaged`) |
| `report.py` | Write a run's `REPORT.md` | `kaggle_train` |
| `compare_runs.py` | Fair comparison of evaluated runs → `COMPARISON.md` (ADR-021) | `kaggle_train` |
| `training_curves.py` | Print or export training and validation curves | — |
| `run_study.py` | Run a multi-seed study within a time budget (ADR-022, ADR-023) | `kaggle_study` |
| `analyze_study.py` | Write `STUDY.md` + `STUDY.json` from run directories | `kaggle_study`, `reproduce_results` |
| `pilot_report.py` | Pilot gate report `PILOT.md` (ADR-023 §8) | `kaggle_study` (cell 5b) |
| `kaggle_validate.py` | GPU-session validation checks and JSON report | `kaggle_validate` |
| `package_for_kaggle.py` | Build the offline code and data datasets (ADR-016) | — |
| `kaggle_publish_runs.py` | Publish run directories to the private `aiayn-runs` dataset (ADR-020) | `kaggle_train`, `kaggle_study` |

Every script defines `main()` and an `if __name__ == "__main__":` guard (required on Windows, where DataLoader workers are spawned).
Contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#scripts).
