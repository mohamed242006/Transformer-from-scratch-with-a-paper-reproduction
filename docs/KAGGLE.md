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
1. **Session 1:** set `EXPERIMENT` and `RUN_NAME`, keep `RESUME_FROM = None`, then *Save Version → Save & Run All (Commit)*. The data is downloaded and checksum-verified, the tokenizer is trained (its fingerprint is stored in the run), and training stops cleanly at the 11.5 h session budget if needed (exit code 3).
2. **Session 2+:** add the previous version's **output** as an input and set `RESUME_FROM = "/kaggle/input/<output>/runs/<RUN_NAME>"`. The run directory is copied to `/kaggle/working`, its config hash and tokenizer fingerprint are verified, and training continues from the exact update, batch and random state.
3. Each session's output (`/kaggle/working/runs/<RUN_NAME>/`) holds `config.yaml`, `metadata.json` (one entry per session), `metrics.jsonl`, `tokenizer/`, and `checkpoints/{last,last.prev,best}.pt` + `avg/`.

Equivalent commands:
```bash
python scripts/train.py --experiment configs/experiment/multi30k_tiny.yaml --run-dir /kaggle/working/runs/multi30k_tiny_s1 --prepare-data
python scripts/train.py --resume /kaggle/input/<output>/runs/multi30k_tiny_s1 --run-dir /kaggle/working/runs/multi30k_tiny_s1 --prepare-data
```
Only `train.{max_updates, max_epochs, time_budget_minutes, log_every, loggers, device}` may change on resume (ADR-018); any other change is refused.

