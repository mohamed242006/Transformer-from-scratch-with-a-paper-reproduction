# Running on Kaggle

How to get this project onto Kaggle and validate a GPU session before training. Background: [ADR-010](decisions/ADR-010-compute-platform.md) (why Kaggle), [ADR-016](decisions/ADR-016-kaggle-code-and-data-delivery.md) (how code and data get there), [ADR-013](decisions/ADR-013-run-artifacts-and-resumption.md) (run artifacts), [kaggle_validation_notebook.md](kaggle_validation_notebook.md) (check design).

| Piece | Path |
|---|---|
| Validation logic (tested) | `src/aiayn/diagnostics/` |
| Validation CLI | `scripts/kaggle_validate.py` |
| Packaging CLI | `scripts/package_for_kaggle.py` |
| Notebook (thin wrapper) | `notebooks/kaggle_validate.ipynb` |

---

## 0. Quick start: GitHub clone (current workflow)
Kaggle notebook, **Accelerator: GPU T4**, **Internet: ON**. Either import `notebooks/kaggle_validate.ipynb` (default `CODE_SOURCE = "github"`) and *Run All*, or run these cells:
```python
!git clone --depth 1 --branch master https://github.com/mohamed242006/Transformer-from-scratch-with-a-paper-reproduction.git /kaggle/working/aiayn
%cd /kaggle/working/aiayn
torch_before = !python -c "import torch; print(torch.__version__)"
!pip install -q -e . --no-deps
!pip install -q "sacrebleu>=2.4"
!python -m pytest tests/policy tests/parity tests/numerical -q -p no:cacheprovider
!python scripts/kaggle_validate.py --data-config configs/data/multi30k.yaml --model-config configs/model/transformer_tiny.yaml --model-config configs/model/transformer_iwslt.yaml --data-override translation.root=/kaggle/working/data --train-tokenizer-if-missing --retrain-tokenizer-check --expected-fingerprint f04bec5925700e0aedbfd3e700e67823b3a59f0bdd17d5e01ab7a9f9f07c930a --expected-torch-version {torch_before[0]} --require-cuda --check-training-resume --output /kaggle/working/validation/report.json
```
Multi30k is downloaded from its pinned URLs (checksum-verified). The tokenizer is trained on Kaggle and must reproduce the local fingerprint (`tokenizer.fingerprint`). The last line prints the report table and exits non-zero on any critical failure. The offline alternative (Kaggle Datasets, no internet) is described in §2–4.

## 1. One-time Kaggle setup
1. Create a Kaggle account and **verify your phone number** (*Settings → Phone verification*). Without it, notebooks get neither GPUs nor internet.
2. Check your weekly GPU quota on your profile; it is about 30 h, and a validation run uses about 5 min.
3. *(Optional, for command-line uploads)* create an API token: *Settings → API → Create New Token*. This downloads `kaggle.json`. Put it in `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`, then `pip install kaggle`.

## 2. Package the datasets (on your machine)
Prerequisites: Multi30k prepared and the tokenizer trained locally (both done in Week 2):
```bash
python scripts/prepare_data.py    --config configs/data/multi30k.yaml   # if data/raw is empty
python scripts/train_tokenizer.py --config configs/data/multi30k.yaml   # if data/tokenizers is empty
python scripts/package_for_kaggle.py --kaggle-user <your-kaggle-username>
```
Output (in `dist/`, never shipped inside the code bundle itself):
```
dist/kaggle/
├── aiayn-code/
│   ├── dataset-metadata.json
│   └── aiayn/                     project tree + AIAYN_MANIFEST.json (tree hash = code version)
└── aiayn-multi30k/
    ├── dataset-metadata.json
    ├── train|val|test_2016_flickr.{en,de}.gz    pinned raw archives (checksums re-verified)
    ├── multi30k_bpe8000.model                   tokenizer
    ├── tokenizer_fingerprint.txt                expected on Kaggle (cross-machine check)
    └── SHA256SUMS
```
The script prints the code **tree hash**. Every validation report shows the same value under `environment.code_tree_sha256`, so you can confirm Kaggle ran the code you packaged.

## 3. Upload the datasets (keep them **private**)
**Option A: Kaggle CLI** (from the project root):
```bash
kaggle datasets create  -p dist/kaggle/aiayn-code --dir-mode zip      # first time
kaggle datasets create  -p dist/kaggle/aiayn-multi30k
kaggle datasets version -p dist/kaggle/aiayn-code --dir-mode zip -m "code <tree-hash-prefix>"   # after code changes
```
New datasets created by the CLI are private by default.

**Option B: web UI.** *Datasets → New Dataset*. Drag the **contents** of `dist/kaggle/aiayn-code/` (the `aiayn/` folder), name it `aiayn-code`, keep it *Private*, create. Repeat with the files of `dist/kaggle/aiayn-multi30k/` as `aiayn-multi30k`. After code changes, open the dataset → *New Version* → upload again.

**Why private:** Multi30k (Flickr30k captions) and IWSLT (TED, CC BY-NC-ND) must not be redistributed publicly, and the code is unpublished work.

## 4. Create and run the validation notebook
1. *Code → New Notebook → File → Import Notebook* → upload `notebooks/kaggle_validate.ipynb`.
2. Right-hand panel → *Session options*: **Accelerator: GPU T4** (single or ×2; one GPU is used). **Internet:** optional. Persistence: off is fine.
3. *Add Input* → your `aiayn-code` and `aiayn-multi30k` datasets. They appear under `/kaggle/input/<slug>`. If Kaggle mounts them elsewhere, edit `CODE_DATASET` / `DATA_DATASET` in cell 0 (the notebook searches recursively below those folders).
4. **Run All** for an interactive check, or *Save Version → Save & Run All (Commit)* to keep `/kaggle/working/validation/report.json` as a versioned output.
5. Success is the last cell printing `VALIDATION PASSED`. Any critical failure raises with a table explaining it.

### What runs (target < 10 min; expected ≈ 3–5 min on a T4)
| Cell | Content | Expected time |
|---|---|---|
| 1 | Locate the code bundle (folder or zip), copy it to `/kaggle/working/aiayn` | < 10 s |
| 2 | `pip install -e . --no-deps --no-build-isolation`; install only *missing* deps; torch never touched | 10–60 s |
| 3 | `nvidia-smi`, versions | < 5 s |
| 4 | Preflight: `pytest tests/policy tests/parity tests/numerical` (93 tests) | ≈ 30–60 s |
| 5 | `kaggle_validate.py`: env, deps (+ torch unchanged), CUDA (arch supported, GPU-vs-CPU-fp64 numerics, fp16), data (offline mirror + checksums), tokenizer (fingerprint + retrain determinism), loader, then per model: build (exact parameter count), forward (initial CE band), backward (fp16 + GradScaler, all gradients finite), **VRAM** (worst-case batch, measured), **throughput** (30 timed steps) | ≈ 1.5–3 min |
| 6 | Summary table + memory/throughput details | < 5 s |

### Troubleshooting
| Failing check | Cause | Fix |
|---|---|---|
| `cuda.available` (arch not supported) | The PyTorch build lacks this GPU's architecture (typical for P100 / sm_60 on recent builds) | Switch the accelerator to T4 |
| `deps.torch_unchanged` | pip replaced Kaggle's torch | Re-run; only `--no-deps` installs are allowed |
| Cell 4 fails | PyTorch numerics differ from the local version | Stop; record versions; reproduce locally (ADR-004) |
| `data.prepare` checksum error | Archive in the dataset corrupted or replaced | Re-package and re-upload; never edit pinned checksums |
| `tokenizer.load` fingerprint mismatch | Different tokenizer than the one packaged | Re-upload the data bundle; check `sentencepiece` versions |
| `tokenizer.deterministic` fails | SentencePiece version differs across machines | Use the bundled tokenizer only (never retrain on Kaggle); record versions |
| `model[*].memory` warn | Peak reserved > 90% of GPU memory | Lower `batching.max_tokens` (see `suggested_max_tokens_80pct`) |

## 5. Measured VRAM and throughput (Tesla T4, 2026-11-16)
First validation run on Kaggle: Tesla T4, Python 3.13, fp16 autocast + GradScaler, Multi30k data (V = 8,000). **VRAM** is the peak allocated by PyTorch for one **worst-case** training step (forward, backward, Adam update): 40 sentences × 100 tokens per side = 4,000 padded tokens per side at `max_tokens = 4096`. **Throughput** is the mean over 30 real training steps at `max_tokens = 4096`.

| Model | Params | **Peak VRAM (measured)** | Earlier CPU-based estimate | **Throughput (measured)** |
|---|---|---|---|---|
| `transformer_tiny` | 7.58 M | **898 MiB** | 1,084 MiB (+21%) | **78,808 target tokens/s** |
| `transformer_iwslt` | 35.6 M | **2,243 MiB** | 3,618 MiB (+61%) | **24,669 target tokens/s** |

The estimates were conservative (they used bf16 autocast on CPU as a stand-in for fp16 on CUDA). The measured values are authoritative.

**Memory headroom** (T4 ≈ 15.1 GB usable, budget 80% ≈ 12.1 GB). The static part (fp32 weights + gradients + 2 Adam moments ≈ 16 bytes/parameter) is ≈ 116 MiB for tiny and ≈ 544 MiB for iwslt. Activations grow linearly with tokens per step, so the largest worst-case token budgets within 80% are roughly:

| Model | Activations per 4,096 tokens | Max `max_tokens`/side at 80% |
|---|---|---|
| `transformer_tiny` | ≈ 782 MiB | ≈ 60k |
| `transformer_iwslt` | ≈ 1,700 MiB | ≈ 28k |

So even the paper's 25k tokens per side fits on one T4 for the IWSLT model. Memory is not the constraint; **time is.**

## 6. GitHub vs Kaggle Dataset for delivering the code
| | Kaggle Dataset (chosen, ADR-016) | GitHub clone |
|---|---|---|
| Needs a git repository | No | Yes (currently removed) |
| Works with internet off | **Yes** | No |
| Privacy | Private by default | Public repo exposes unfinished work, unless private (then the notebook needs a token) |
| Code identity | Tree hash in `AIAYN_MANIFEST.json` | Commit SHA |
| Update cost | Re-package + `kaggle datasets version` (≈ 1 min) | `git push`, then the notebook pulls |
| History | Dataset versions (snapshots) | Full line-level history |
| Portfolio visibility | None | High |

**Update 2026-11-16:** the owner restored the GitHub repository (`master`), so the **GitHub clone is now the primary path** (§0); Kaggle Datasets remain the offline fallback. The comparison below is kept for reference.

**Original recommendation (no repository at the time):** use **Kaggle Datasets now**. They are the only option that works without a repository and without internet, keep licensed data private, and the tree hash gives the provenance that a commit SHA would. Add GitHub later, when you want public visibility for the portfolio: a public repo is valuable on a CV, and the notebook can switch to `git clone` with a two-line change. Keep the data bundle on Kaggle either way.

## 7. Recommended training settings on a Kaggle T4
| Setting | Multi30k + `transformer_tiny` | IWSLT14 + `transformer_iwslt` |
|---|---|---|
| Experiment / train config | `multi30k_tiny` / `multi30k_t4` | (Week 8) / `iwslt_t4` |
| AMP | **fp16 autocast + GradScaler** (T4 has no bf16; ADR-014) | same |
| Token budget (`data.batching.max_tokens`) | **4,096** per side (≈ 260 sentences) | **8,192** per side (≈ 300 sentences) |
| Gradient accumulation (`update_freq`) | **1** | **1** (≈ 8k-token updates, the usual size for IWSLT) |
| Updates per epoch | ≈ 128 | ≈ 500 |
| Learning rate | inverse-sqrt, peak 7e-4, warmup 400 | inverse-sqrt, peak 7e-4, warmup 4,000 |
| Length | 30 epochs ≈ 3,800 updates | 30,000 updates ≈ 60 epochs |
| Validation / checkpoints | every epoch / every 500 updates | every 1,000 updates / every 1,000 updates |
| Worst-case peak VRAM | ≈ 0.9 GB | ≈ 4 GB |

Why not larger batches? Multi30k is small (≈ 0.46 M target tokens per epoch): bigger batches would mean very few updates per epoch. For IWSLT, 8k tokens is the common setting, and the paper's 25k only pays off with far more data. `paper_base` (25k via 8,192 × 3 accumulation, exact Noam) is kept for reference.

### Estimated training time (T4, from the measured throughput)
| Run | Compute | + validation, checkpoints, setup | **Per run** | Planned runs |
|---|---|---|---|---|
| Multi30k, tiny, 30 epochs | 30 × 0.46 M tokens ÷ 78.8k tok/s ≈ **3 min** | data prep + tokenizer ≈ 1 min, I/O ≈ 1 min | **≈ 5–8 min** | ablations: 8 variants × 3 seeds ≈ 24 runs ≈ **2.5–3 GPU-h** |
| IWSLT14, iwslt, 30k updates | ≈ 60 × 4 M tokens ÷ ≈ 22k tok/s ≈ **3 h** | validation ≈ 5 min, 30 checkpoints of ≈ 0.44 GB ≈ 2 min | **≈ 3–3.5 h** | headline: 3 seeds ≈ **10–11 GPU-h** (one session each) |

Assumptions: IWSLT14 DE→EN ≈ 160k pairs ≈ 4 M target tokens per epoch at a 10k vocabulary (to be measured in Week 7). IWSLT throughput is taken slightly below the Multi30k measurement (longer sentences, larger output layer). Each IWSLT run fits in one 12 h session, and runs that don't finish resume exactly (ADR-013). Total plan ≈ 15 GPU-h, about half of one week's Kaggle quota.

## 8. Training on Kaggle (multi-session, resumable)
Notebook `notebooks/kaggle_train.ipynb` (thin wrapper around `scripts/train.py`):
1. **Cell 0:** list the runs in `RUNS` as `(experiment config, run name)` pairs. The first run is the comparison reference. `OVERRIDES` applies to new runs only, for example `["train.seed=2"]` with `*_s2` run names.
2. **Every session:** *Save Version → Save & Run All (Commit)*. The data is downloaded and checksum-verified, and the tokenizer is trained, with its fingerprint stored in the run. Each run gets the time left in the session, and training stops cleanly at the budget if needed (exit code 3).
3. **Resuming is automatic.** Attach the previous version's output or the `aiayn-runs` dataset as input. Each run in `RUNS` is resumed from the most advanced copy found under `/kaggle/input/**` (§9). Its config hash and tokenizer fingerprint are verified, and training continues from the exact update, batch and random state.
4. Each session's output (`/kaggle/working/runs/<run name>/`) holds `config.yaml`, `metadata.json` (one entry per session), `metrics.jsonl`, `tokenizer/`, and `checkpoints/{last,last.prev,best}.pt` + `avg/`.

Equivalent commands:
```bash
python scripts/train.py --experiment configs/experiment/multi30k_tiny.yaml --run-dir /kaggle/working/runs/multi30k_tiny_s1 --prepare-data
python scripts/train.py --resume /kaggle/input/<output>/runs/multi30k_tiny_s1 --run-dir /kaggle/working/runs/multi30k_tiny_s1 --prepare-data
```
Only `train.{max_updates, max_epochs, time_budget_minutes, log_every, loggers, device}` may change on resume (ADR-018); any other change is refused.

### Evaluation (Week 6)
The training notebook's last cell evaluates the test set after a completed run:
```bash
python scripts/evaluate.py --run-dir /kaggle/working/runs/multi30k_tiny_s1                         # best, beam 4
python scripts/evaluate.py --run-dir /kaggle/working/runs/multi30k_tiny_s1 --checkpoint averaged   # last 5 snapshots
python scripts/evaluate.py --run-dir /kaggle/working/runs/multi30k_tiny_s1 --greedy
```
Results are written to `runs/<id>/eval/*.json` (cased sacreBLEU with signature, lowercased BLEU, chrF, BLEU by source length, decode time) and `*.hyp`. For an **already finished** run from an earlier session, attach that output and run the same commands on `/kaggle/input/<output>/runs/<id>` copied to `/kaggle/working`. Evaluation needs no training state, only `config.yaml`, `tokenizer/` and the checkpoints.


## 9. Preserving runs between sessions (ADR-020)
**Lesson learned:** the first Multi30k run was lost because the interactive session ended before a version was saved. On Kaggle, `/kaggle/working` survives only when the notebook runs as **Save Version → Save & Run All (Commit)**.

**One-time setup** (for the second, independent copy):
1. kaggle.com → *Settings → API → Create New Token* (downloads `kaggle.json` with `username` and `key`).
2. In the training notebook: *Add-ons → Secrets* → add `KAGGLE_USERNAME` and `KAGGLE_KEY` with those values, and enable both for the notebook.

**Every session** (`notebooks/kaggle_train.ipynb`; edit only the `RUNS` list of `(experiment, run name)` pairs):
1. *Add Input*: the previous version's output and/or your `aiayn-runs` dataset (if they exist).
2. **Save Version → Save & Run All (Commit).** The notebook then:
   - for each run in `RUNS`, auto-resumes it from the most advanced copy found in `/kaggle/input/**` (or starts it), with the time left in the session as its budget;
   - trains until completion or the 11.5 h budget (exit code 3 = resumable);
   - for a completed run, evaluates the test set (best beam 4, averaged beam 4, best greedy) and writes `runs/<name>/REPORT.md`;
   - publishes every run directory (minus `last.prev.pt`) as a new version of the **private** dataset `<user>/aiayn-runs`, carrying over the other runs already in the attached dataset.
3. Copies that exist after the session: the commit output **and** `aiayn-runs`. Either one is enough to resume or re-evaluate.

| What | Where it is preserved |
|---|---|
| Checkpoints (`last.pt`, `best.pt`, `avg/`) | commit output + `aiayn-runs` |
| Run directory (`config.yaml`, `metadata.json`, `metrics.jsonl`, `tokenizer/`) | commit output + `aiayn-runs` |
| Evaluation outputs (`eval/*.json`, `eval/*.hyp`) | commit output + `aiayn-runs` |
| BLEU report (`REPORT.md`) | commit output + `aiayn-runs` |

Manual equivalents: `python scripts/report.py --run-dir <run>` and `python scripts/kaggle_publish_runs.py --run <run> --user <you> [--previous /kaggle/input/aiayn-runs] [--dry-run]`.

## 10. Baselines and the comparison session (Week 7, ADR-021)
The default `RUNS` list trains the three RNN baselines after the Transformer reference, in one committed session:

| Run name | Experiment | Model |
|---|---|---|
| `multi30k_tiny_s1` | `multi30k_tiny` | Transformer (reference; found finished in `aiayn-runs`, so only missing evaluations run) |
| `multi30k_bilstm_luong_s1` | `multi30k_bilstm_luong` | BiLSTM + Luong attention |
| `multi30k_gru_bahdanau_s1` | `multi30k_gru_bahdanau` | BiGRU + Bahdanau attention |
| `multi30k_lstm_seq2seq_s1` | `multi30k_lstm_seq2seq` | LSTM, no attention |

**Attach the `aiayn-runs` dataset** so the reference is reused instead of retrained. Once the reference and at least one baseline are evaluated, the notebook writes `runs/COMPARISON.md` (also copied into the reference run, so it is published). The file contains:
- the fairness checks;
- BLEU with 95% CI, ΔBLEU and a paired-bootstrap p-value against the Transformer;
- chrF, length ratio, validation NLL, wall time, throughput and memory.

Locally: `python scripts/compare_runs.py --run runs/multi30k_tiny_s1 --run runs/multi30k_bilstm_luong_s1 ... --out runs/COMPARISON.md`. The first run is the reference. Exit code 1 means a fairness check failed.

More seeds: add `("configs/experiment/<exp>.yaml", "<exp>_s2")` entries with `OVERRIDES = ["train.seed=2"]`. Overrides apply to new runs only. The comparison adds a mean ± std table.

## 11. The Week 8 study (ADR-022)
`notebooks/kaggle_study.ipynb` with `STUDY = "configs/study/week8.yaml"` and `REFERENCE_RUN = "multi30k_tiny_s1"` runs the Week 8 study. That is about 110 runs: 5-seed main comparison, decoding grid, training sensitivity, tuned configs and Transformer ablations. Expect about 9 T4-hours, so one or two sessions.

**Each session:**
1. Use the same secrets as §9, GPU T4, Internet on, and **add the `aiayn-runs` dataset** as input. It holds the Week 7 seed-1 runs, which are reused.
2. **Save Version → Save & Run All (Commit).** The study runner:
   - reuses every finished run it finds;
   - works through the plan in priority order;
   - stops about 40 minutes before the session limit;
   - writes `runs/STUDY.md` and `STUDY.json`, also copied into `multi30k_tiny_s1/`;
   - publishes all study runs, each slimmed to ~30–60 MB, as a new `aiayn-runs` version.
3. If the notebook prints `run again to continue` (exit 3), wait until the new `aiayn-runs` version shows as ready, then repeat with it attached. The study is done when `STUDY.md` says **complete**.

**Locally:**
- `python scripts/run_study.py --study configs/study/week8.yaml --work runs --dry-run` lists the remaining tasks.
- `python scripts/analyze_study.py --study configs/study/week8.yaml --work runs --input <downloaded aiayn-runs> --out runs/STUDY.md --prepare-data` rebuilds the report from downloaded runs.

A diverged run (for example at LR ×2) is recorded as `FAILED.json` and reported in the tables. It is never retried, and in the selection it counts as the worst option.

## 12. Week 9: IWSLT'14 (ADR-023)
**Phase 0** (done, no GPU): pinned data, protocols A + B, incremental decoding, study rules, pilot configs.

**Status (2026-11-22): complete.** The final results are in [results.md](results.md) Part I, and the evidence is in `experiments/2026-11-22_iwslt14-week9-benchmark/`. To check the reported numbers without a GPU, run `notebooks/reproduce_results.ipynb` locally. The steps below retrain.

**Phase 1**, in two committed sessions:
1. **Validation (~15 min).** In `notebooks/kaggle_validate.ipynb`, keep the defaults: `DATA_CONFIG = configs/data/iwslt14.yaml` and the three IWSLT models. It downloads and checks the archive, builds the splits (the counts must match), trains the 10k tokenizer (the fingerprint must match `b0c6c22a…`), and measures T4 memory and throughput for the Transformer and both RNNs (gate G4).
2. **Pilot (~10 short runs).** In `notebooks/kaggle_study.ipynb`, set `STUDY = "configs/study/week9_pilot.yaml"` and `REFERENCE_RUN = "iwslt14_pilot_transformer_s1"`. Add as inputs:
   - the `aiayn-runs` dataset;
   - **the validation notebook's output from step 1**, so `PILOT.md` includes its measurements.

   Save with *Save & Run All (Commit)*. Cell 5b writes **`runs/PILOT.md`**, which is also copied into `iwslt14_pilot_transformer_s1/` and published. It contains:
   - validation curves (also as `pilot_curves.csv`);
   - throughput, peak memory and stability for each run;
   - learning-rate selection with grid-edge flags;
   - gates G1–G4 with their evidence;
   - the Post-LN go/no-go decision and the Phase 2 recommendation.

   The cell prints `pilot report exit: 0` for GO, `2` for NO-GO and `3` while runs are unfinished (run the notebook again with the newest `aiayn-runs` attached).

**GRU + Bahdanau memory (ADR-023 Amendment 1).** At 8,192 tokens it runs out of memory on a T4, so it trains with 4,096-token micro-batches × 2 accumulation steps (the same effective update size). Re-validate it before the pilot:
- In `kaggle_validate.ipynb` set `MODEL_CONFIGS = ["configs/model/gru_bahdanau_iwslt.yaml"]` and `VRAM_MAX_TOKENS = 4096`.
- Commit, then check that the memory line says PASS.
- Attach **both** validation outputs (the full one and the GRU-only one) to the pilot. `PILOT.md` merges them, newest first.

**Phase 1c and Phase 2 (done; hyperparameters frozen).** The learning rates selected at full budget (ADR-023 Amendment 3) are already in `configs/experiment/iwslt14_*.yaml`, and nothing needs editing. To rerun the 3-seed benchmark, use `notebooks/kaggle_study.ipynb` with its defaults: `STUDY = "configs/study/week9_benchmark.yaml"`, `REFERENCE_RUN = "iwslt14_transformer_s1"`. Then:
- *Save & Run All (Commit)*, repeated with the newest `aiayn-runs` attached until `STUDY.md` says **complete**;
- the runner trains the 9 runs (3 models × 3 seeds), selects checkpoints and decoding on validation, evaluates on test and writes `STUDY.md` and `STUDY.json`.

**Expected cost on a T4** (estimate, from the T4 throughput in Amendment 1):
- Transformer about 2.7 h per seed;
- BiLSTM + Luong about 1.4 h per seed;
- GRU + Bahdanau not measured on a T4. It was 1.6× slower than the Transformer on the RTX 5070 Ti, so allow about 4–5 h per seed.

That is roughly **25 T4-hours plus evaluation, about 3 sessions**, close to one week's Kaggle GPU quota.

**The reported benchmark was run locally** (one RTX 5070 Ti, 8.9 GPU-hours for all 24 IWSLT'14 runs) with the same configs, via `scripts/run_study.py` (README → "Reproduce the IWSLT'14 benchmark"). Retrained numbers should agree within seed noise, not bit for bit. Two things may differ: the GRU's late divergence (results.md §2.5) may hit different seeds, and the T4 uses fp16 like the local GPU.

Every evaluation reports sacreBLEU (primary) and `bleu_fairseq` (tokenized lowercased, for literature comparison; ADR-023 §4).
