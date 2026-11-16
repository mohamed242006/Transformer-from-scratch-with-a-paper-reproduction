# Kaggle Validation Notebook: Design

- **Status:** **Implemented** (pre-Week-5): `aiayn.diagnostics`, `scripts/kaggle_validate.py`, `notebooks/kaggle_validate.ipynb`. Usage: [KAGGLE.md](KAGGLE.md). Differences from this design: code arrives as a Kaggle Dataset with a tree hash instead of a git commit (ADR-016), and cell 7 (checkpoint loading) was implemented in Week 5 as the `training.resume` check (`--check-training-resume`): a small model trains uninterrupted and interrupted + resumed on the GPU, and the per-update losses are compared. The original design follows for reference.
- **Related:** [ADR-010](decisions/ADR-010-compute-platform.md), [ADR-013](decisions/ADR-013-run-artifacts-and-resumption.md), [ADR-014](decisions/ADR-014-numerical-precision.md), [ADR-004](decisions/ADR-004-parity-testing.md)

## 1. Purpose
Before any GPU-hours are spent on training, prove that a fresh Kaggle session can run this project correctly: the environment, the GPU, the dependencies, the data, the tokenizer, the model and checkpoints. The notebook is run (a) once to validate the platform, (b) at the start of every training session as a preflight (ADR-013 §9), and (c) after any Kaggle image update.

## 2. Design principles
1. **Thin notebook, logic in the repository** (ADR-010 §5). Each cell calls one function from `scripts/kaggle_validate.py` (to be implemented in Week 5). The checks are unit-tested on CPU, and GPU-only checks skip cleanly when CUDA is absent.
2. **Fail fast, in dependency order.** A failed *critical* check stops the notebook (`raise`) before later cells waste quota. *Warnings* are recorded and execution continues.
3. **Offline-capable.** Every check has a path that needs no internet (code and data from `/kaggle/input`).
4. **Machine-readable result.** Every check appends `{name, status: pass|warn|fail, value, expected, seconds}` to `/kaggle/working/validation/report.json`. That file becomes part of the notebook output and is attached to every training run's metadata.
5. **Cheap.** Target is under 8 minutes of GPU session time.

## 3. Kaggle settings and inputs
| Setting | Value |
|---|---|
| Accelerator | GPU T4 (preferred, fp16 tensor cores). P100 is accepted only if check 2.4 passes. |
| Internet | On if available; the notebook must also pass with it off |
| Persistence | Files under `/kaggle/working` |

| Input (Kaggle Dataset or previous notebook output) | Mount | Used by |
|---|---|---|
| `aiayn-code` (repository snapshot at a pinned commit), when there is no internet | `/kaggle/input/aiayn-code` | Cell 3 |
| `multi30k-raw` (the 6 original `.gz` archives) | `/kaggle/input/multi30k-raw` | Cell 4 |
| Previous training output (`runs/<id>/…`), optional | `/kaggle/input/<output>` | Cells 5, 7 |

## 4. Parameters cell (cell 0)
```
RUN_ID            = None | "2026-12-01_transformer-multi30k-postln_s1"
CODE_SOURCE       = "github" | "dataset"
CODE_COMMIT       = "<40-hex sha>"          # must match after checkout
DATA_CONFIG       = "configs/data/multi30k.yaml"
MODEL_CONFIG      = "configs/model/transformer_tiny.yaml"
DATA_MIRROR       = "/kaggle/input/multi30k-raw"
RESUME_DIR        = None | "/kaggle/input/<output>/runs/<RUN_ID>"
EXPECTED_TOKENIZER_FINGERPRINT = "f04bec59…" # recorded locally in Week 2 (multi30k, BPE 8k)
WORK_ROOT         = "/kaggle/working"
```

## 5. Checks (cell by cell)

### Cell 1: Environment
| # | Check | Pass criterion | Severity |
|---|---|---|---|
| 1.1 | Python version | 3.10–3.12 | critical |
| 1.2 | OS, CPU count, RAM | RAM ≥ 12 GB | warn |
| 1.3 | Free disk in `/kaggle/working` | ≥ 5 GB (ADR-013 budget ≈ 1.8 GB per run) | critical |
| 1.4 | Internet reachability (HEAD request, 5 s timeout) | Recorded only; decides code/data paths | info |
| 1.5 | Session start time | Recorded; feeds the trainer's wall-clock budget | info |

### Cell 2: CUDA
| # | Check | Pass criterion | Severity |
|---|---|---|---|
| 2.1 | `torch.cuda.is_available()` | True | critical |
| 2.2 | Device name, total memory | ≥ 15 GB | warn |
| 2.3 | Compute capability | (7, 5) T4 → fp16 AMP. (6, 0) P100 → fp32 plan (ADR-014). | info |
| 2.4 | **Build supports this architecture:** `f"sm_{major}{minor}" in torch.cuda.get_arch_list()` | True (catches PyTorch builds that dropped Pascal) | critical |
| 2.5 | `torch.version.cuda`, cuDNN version, `nvidia-smi` | Recorded | info |
| 2.6 | GPU matmul fp32 vs CPU fp64 reference | max error < 1e-4 | critical |
| 2.7 | fp16 autocast matmul + softmax | Finite, error < 1e-2 | critical (on T4) |

### Cell 3: Code and dependency installation
| # | Check | Pass criterion | Severity |
|---|---|---|---|
| 3.1 | Get the code: `git clone` + `git checkout CODE_COMMIT`, or copy from `/kaggle/input/aiayn-code` | `git rev-parse HEAD == CODE_COMMIT` (or the snapshot manifest matches) | critical |
| 3.2 | Record the torch version **before** installing | — | info |
| 3.3 | `pip install -e . --no-deps`, then install only *missing* runtime dependencies (sentencepiece, sacrebleu, pyyaml) at the pyproject bounds | Exit code 0 | critical |
| 3.4 | Torch version **after** installing | Unchanged (pip must never replace Kaggle's CUDA build) | critical |
| 3.5 | `pip check`, `import aiayn`, `aiayn.__version__` | Clean | critical |
| 3.6 | `pytest -m "parity or numerical" -q` + `tests/policy` | All pass (catches PyTorch drift, ADR-004) | critical |
| 3.7 | **GPU parity smoke:** `MultiHeadAttention` and `LayerNorm` outputs on CUDA (fp32) vs CPU (fp64) with the same weights | ≤ 1e-5 | critical |

### Cell 4: Dataset loading
| # | Check | Pass criterion | Severity |
|---|---|---|---|
| 4.1 | `prepare_data.py --config DATA_CONFIG --mirror-dir DATA_MIRROR --override translation.root=/kaggle/working/data` | SHA-256 verified for all 6 files | critical |
| 4.2 | Split sizes | 29,000 / 1,014 / 1,000 aligned pairs | critical |
| 4.3 | Build train dataset + DataLoader (`num_workers` 0 and 2) | Every index exactly once; identical batches across worker counts | critical |
| 4.4 | Padding efficiency, one-epoch iteration time | Efficiency ≥ 0.9 on train; time recorded | warn |
| 4.5 | Copy task dataset | Builds; masks valid | critical |

### Cell 5: Tokenizer loading
| # | Check | Pass criterion | Severity |
|---|---|---|---|
| 5.1 | With `RESUME_DIR`: load `runs/<id>/tokenizer/*.model`. Otherwise train it with `train_tokenizer.py`. | Loads; special IDs = (0, 1, 2, 3) | critical |
| 5.2 | **Cross-machine fingerprint:** `tokenizer.fingerprint == EXPECTED_TOKENIZER_FINGERPRINT` (or the value in `metadata.json`) | Equal (validates the Week 2 content-hash design across machines) | critical |
| 5.3 | Round trip on 100 validation sentences | 100% exact | critical |
| 5.4 | Vocabulary size | Equals config `vocab_size` | critical |

### Cell 6: Model construction
| # | Check | Pass criterion | Severity |
|---|---|---|---|
| 6.1 | `build_transformer(MODEL_CONFIG, vocab_size, pad_id, seed)` | Builds | critical |
| 6.2 | `count_parameters == expected_parameter_count` | Exact (e.g. 7,577,600 for tiny at V = 8000) | critical |
| 6.3 | Tying | `output_weight is src_embedding.weight` | critical |
| 6.4 | Forward on a real Multi30k batch on CUDA, fp32, `debug_checks=True` | Logits `[B, T, V]` finite; CE at init in `[ln V, ln V + 1.5]` | critical |
| 6.5 | Same batch under fp16 autocast (T4) | Finite; max |Δlogits| vs fp32 < 5e-2 | critical (T4) |
| 6.6 | Backward pass, fp16 autocast + `GradScaler` | All grads finite; no parameter without a gradient | critical |
| 6.7 | Micro-benchmark: 20 train steps at the config's `max_tokens` | tokens/s and peak memory recorded; peak < 85% of GPU memory | warn |

### Cell 7: Checkpoint loading (ADR-013 format; implemented with the Week 5 trainer)
| # | Check | Pass criterion | Severity |
|---|---|---|---|
| 7.1 | Save → load round trip in `/kaggle/working/validation/` (config + state, `torch.load(weights_only=True)`) | Identical logits on GPU | critical |
| 7.2 | Cross-device: a CPU-made state dict (shipped in the code snapshot as a tiny fixture) loads on CUDA | Logits within 1e-5 of the stored CPU reference | critical |
| 7.3 | With `RESUME_DIR`: load `last.pt` (fall back to `last.prev.pt`) | Loads with `weights_only=True` | critical |
| 7.4 | Resume compatibility: config hash, tokenizer fingerprint, `TokenBudgetBatchSampler.load_state_dict` | All match / no refusal | critical |
| 7.5 | Optimizer, scaler and RNG states present; step > 0 | Present | critical |
| 7.6 | Copy `RESUME_DIR` → `/kaggle/working/runs/<id>` and check free space afterwards | Copy verified (sizes + SHA-256 of `last.pt`) | critical |

### Cell 8: Report
Write `report.json` with an environment fingerprint (Python, torch, CUDA, GPU, commit, package versions). Print a pass/warn/fail table. `raise` if any critical check failed. Training notebooks run cells 1–7 as their first step and refuse to start otherwise.

## 6. Failure playbook
| Failure | Likely cause | Action |
|---|---|---|
| 2.4 | PyTorch build without sm_60 (P100) | Switch the accelerator to T4 |
| 3.4 | pip replaced torch | Reinstall with `--no-deps`; never let dependencies resolve torch |
| 3.6 / 3.7 | PyTorch numerics changed | Stop; reproduce locally with the same torch version; review ADR-004 |
| 4.1 | Mirror archive corrupted or replaced | Re-upload the original archives; never edit checksums to match |
| 5.2 | Tokenizer differs across machines | Stop; compare `sentencepiece` versions; retrain only under a new run ID |
| 6.5 / 6.6 | fp16 overflow | Check ADR-014 rules; fall back to fp32 for that run and log it |
| 7.3 | Truncated `last.pt` (session killed mid-write) | Use `last.prev.pt` (ADR-013 §3) |

## 7. Estimated cost
Cells 1–3 ≈ 2–3 min (most of it the test suite), cell 4 ≈ 30 s, cell 5 ≈ 30 s (+1 min if training the tokenizer), cell 6 ≈ 1 min, cell 7 ≈ 1 min. **Total ≈ 5–7 min.**

## 8. Implementation plan (Week 5)
1. `scripts/kaggle_validate.py`: one function per check returning a `CheckResult`; a CLI `--only cell4`; JSON report writer.
2. `tests/unit/test_kaggle_validate.py`: CPU tests with CUDA checks mocked or skipped.
3. `notebooks/kaggle_validate.ipynb`: parameters cell + one call per cell, outputs cleared.
4. A tiny CPU-generated state-dict fixture (< 100 KB) for check 7.2.
