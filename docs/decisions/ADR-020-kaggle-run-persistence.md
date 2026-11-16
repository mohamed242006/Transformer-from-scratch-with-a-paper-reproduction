# ADR-020: Preserve Kaggle runs with commit-mode sessions and a private runs dataset

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** Lead engineer, project owner
- **Related:** [ADR-010](ADR-010-compute-platform.md), [ADR-013](ADR-013-run-artifacts-and-resumption.md), [ADR-016](ADR-016-kaggle-code-and-data-delivery.md), [KAGGLE.md §9](../KAGGLE.md)

## Context
The first full Multi30k run (`20261116-175603_multi30k_tiny_s1`) trained successfully on a T4 but was **lost**: the session ended before the outputs were saved as a notebook version, so `/kaggle/working` (checkpoints, run directory) disappeared. Only the console log survived, so no BLEU could be computed. ADR-013 defined *what* a run directory contains, but not *how it leaves the session*. On Kaggle, `/kaggle/working` is kept only when the notebook runs via **Save & Run All (Commit)**. Interactive sessions lose it when they end.

## Decision
1. **Every training run is launched as a committed notebook version** (`notebooks/kaggle_train.ipynb`, non-interactive end to end): train → evaluate → `REPORT.md` → publish. The commit's output holds `/kaggle/working/runs/<name>/`.
2. **Second, independent copy:** at the end of each session the notebook publishes the run to a **private Kaggle Dataset** (`<user>/aiayn-runs`) through the Kaggle CLI, using the notebook secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`. A dataset version replaces the previous content, so runs already in the attached dataset are carried over. Excluded from upload: `last.prev.pt` and partial files (`*.tmp`, `*.new`, `*.part`).
3. **Stable run names** (`RUN_NAME`, e.g. `multi30k_tiny_s1`) replace timestamped names in notebooks, so a run can be found again.
4. **Auto-resume:** the notebook searches all attached inputs (`/kaggle/input/**`, extracting zips if needed) for `RUN_NAME` and resumes the copy with the most updates (`latest_run`). A finished run "resumes" straight to evaluation.
5. **Evaluation and report are part of the same session** for completed runs (best beam 4, averaged beam 4, best greedy) and are skipped when results already exist.
6. The session budget (11.5 h, ADR-013) guarantees training stops, and the publish step runs, before Kaggle's 12 h limit.

## Consequences
**Positive:** a closed browser, an expired session or a forgotten "Save Version" can no longer lose a finished run; evaluation outputs and BLEU reports travel with the run; multi-session runs need only "attach input, Run All". **Negative:** requires a Kaggle API token as a notebook secret (set once); an upload per session (≈ 0.1 GB tiny, ≈ 1.5 GB iwslt); the dataset accumulates every run (older versions are kept by Kaggle). **Risk:** the dataset is a single growing artifact → it is private, versioned, and runs can be pruned by publishing without them.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Rely on remembering "Save Version" | That is exactly how the first run was lost |
| Hugging Face Hub / Google Drive sync | Extra accounts and tokens outside Kaggle; data-licence concerns for public repos |
| Upload after every checkpoint | Uploads are slow (minutes for iwslt); end-of-session uploads plus the commit output give two copies at negligible cost |
| Keep timestamped run names | Cannot be found automatically by the next session |

## Compliance
`tests/unit/test_kaggle_runs.py` (discovery incl. zips, latest copy wins, staging excludes fallbacks and keeps previous runs, credentials detection, version→create fallback with a fake CLI), `tests/integration/test_evaluation_pipeline.py` (report generation).
