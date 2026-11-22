# notebooks/

Entry points for the project's workflows. Notebooks are **thin wrappers**: every cell calls a script in `scripts/` or a function in `aiayn`, and no model, training or evaluation logic lives here.

## Which notebook do I open?

| Order | Notebook | Runs on | What it does | Time |
|---|---|---|---|---|
| 1 | [`reproduce_results.ipynb`](reproduce_results.ipynb) | **Local CPU** | Regenerates the final IWSLT'14 `STUDY.md` and `STUDY.json` from the committed evidence and checks they are byte-identical. Re-scores all 36 test hypothesis files with sacreBLEU. No GPU, no checkpoints, no training | a few minutes |
| 2 | [`kaggle_validate.ipynb`](kaggle_validate.ipynb) | Kaggle T4 | Checks a GPU session before training: environment, preflight tests, data download and split counts, tokenizer fingerprint, models, VRAM, throughput, exact resume | 5–15 min |
| 3 | [`kaggle_train.ipynb`](kaggle_train.ipynb) | Kaggle T4 | Trains a list of runs with auto-resume, evaluates them on test, writes `REPORT.md` and `COMPARISON.md`, publishes them. Default: the Week 7 Multi30k baselines | 1 session |
| 4 | [`kaggle_study.ipynb`](kaggle_study.ipynb) | Kaggle T4 | Runs a multi-seed study end to end: train, select on validation, evaluate, write `STUDY.md`, publish. Default: the final IWSLT'14 benchmark | several sessions |

**Start with notebook 1** if you want to check the reported numbers. Notebooks 2 → 3 → 4 are for retraining, in that order.

## Workflow → entry point

| Workflow | Notebook | Terminal equivalent |
|---|---|---|
| Dataset preparation | 1 (IWSLT'14), 2 (any data config); 3 and 4 prepare data automatically (`--prepare-data`) | `scripts/prepare_data.py`, `scripts/train_tokenizer.py` |
| Training (single runs) | 3 | `scripts/train.py` |
| Evaluation (test BLEU, chrF, tok-lc BLEU) | 3, 4 (automatic after training) | `scripts/evaluate.py` |
| Two-run comparison with fairness checks | 3 | `scripts/compare_runs.py` |
| Benchmark / study execution | 4 | `scripts/run_study.py` |
| Study analysis (`STUDY.md`, `STUDY.json`) | 4 (from runs), 1 (from the committed archive) | `scripts/analyze_study.py` |
| Pilot gate report (`PILOT.md`) | 4, cell 5b | `scripts/pilot_report.py` |
| Result reproduction without training | 1 | README → "Reproduce the IWSLT'14 benchmark" |

**Terminal only, by design.** These are small utilities, not part of reproducing a result:
- `scripts/translate.py` and `scripts/average_checkpoints.py`: need a trained checkpoint;
- `scripts/inspect_data.py`, `scripts/count_params.py`, `scripts/training_curves.py`, `scripts/report.py`: inspection;
- `scripts/package_for_kaggle.py`, `scripts/kaggle_publish_runs.py`, `scripts/kaggle_validate.py`: called by the Kaggle notebooks or used to package offline data.

## Reproducing each result

| Result | Without training | By retraining |
|---|---|---|
| **IWSLT'14 benchmark** (results.md Part I) | Notebook 1: exact, byte-identical `STUDY.md` | Notebook 4 with `STUDY = "configs/study/week9_benchmark.yaml"` (the default). Or locally: README → "Reproduce the IWSLT'14 benchmark". Expect results within seed noise, not bit-identical (GPU non-determinism) |
| **Week 9 pilot** (`PILOT.md`) | Archived report: `experiments/2026-11-22_iwslt14-week9-benchmark/study/PILOT.md` | Notebook 4 with `STUDY = "configs/study/week9_pilot.yaml"`, `REFERENCE_RUN = "iwslt14_pilot_transformer_s1"` |
| **Multi30k study** (results.md Part II) | **Not possible from the repository.** The run evidence is in the private `aiayn-runs` Kaggle dataset (ADR-020) | Notebook 4 with `STUDY = "configs/study/week8.yaml"`, `REFERENCE_RUN = "multi30k_tiny_s1"` (about 9 T4-hours). Week 7 baselines: notebook 3 with its default `RUNS` |

## Environment assumptions
- **Local notebook (1):**
  - the package installed with `pip install -e .` (README → Development setup);
  - internet once for the IWSLT'14 archive (pinned URL and SHA-256 in `configs/data/iwslt14.yaml`);
  - it can be opened from `notebooks/` or the repository root; it finds the root itself;
  - it writes only to the gitignored `data/` and `runs/reproduce_week9/`.
- **Kaggle notebooks (2–4):**
  - Accelerator **GPU T4**, Internet **ON**;
  - they clone this repository from GitHub into `/kaggle/working/aiayn` (`GITHUB_URL`, `GITHUB_BRANCH` in cell 0) and never reinstall Kaggle's PyTorch;
  - launch notebooks 3 and 4 with **Save Version → Save & Run All (Commit)**, because interactive sessions lose `/kaggle/working`;
  - to keep runs between sessions, add the Kaggle secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` and attach your `aiayn-runs` dataset as input.

  Full setup: [docs/KAGGLE.md](../docs/KAGGLE.md).
- **Checkpoints are never in the repository** (REPOSITORY.md). Notebooks 2–4 create their own; notebook 1 doesn't need any.

## Rules
- Clear outputs before committing.
- Parameters live in cell 0 of each notebook. Hyperparameters live only in `configs/`.

Owner: Research/docs owner. Contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#notebooks).
