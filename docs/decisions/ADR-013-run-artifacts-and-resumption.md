# ADR-013: Run artifacts, checkpointing and resumption on Kaggle

- **Status:** Accepted. **Implemented**: data side in Week 3, trainer side in Week 5 (`aiayn.training.checkpoint`, `aiayn.training.trainer`). Bit-exact resumption is verified on CPU, including interruption after every update; the GPU check is `kaggle_validate.py --check-training-resume`.
- **Date:** 2026-11-15
- **Deciders:** Lead engineer, project owner
- **Related:** [ADR-010](ADR-010-compute-platform.md), [ADR-005](ADR-005-sentencepiece-tokenization.md), [ADR-009](ADR-009-sequence-framing-and-batching.md), [ADR-014](ADR-014-numerical-precision.md)

## Context
All training happens on Kaggle (ADR-010). Constraints that shape every artifact decision:

| Kaggle constraint | Consequence |
|---|---|
| GPU sessions end after ≤ 12 h; weekly GPU quota (~30 h) | Runs span sessions, so exact resumption is mandatory |
| `/kaggle/working` persists only as the saved notebook **output** (≤ ~20 GB) | All artifacts must live under one exportable directory within that budget |
| `/kaggle/input` (datasets, earlier outputs) is **read-only** | Resume = copy the previous run directory from input to working, then continue |
| Internet is off unless enabled (and needs a verified account) | Data, code and tokenizer must be loadable offline |
| Preinstalled PyTorch/CUDA versions change over time | Every session verifies numerics before training |
| Sessions can be killed abruptly | Checkpoint writes must be atomic; a crash must never corrupt the last good state |

## Decision
1. **One self-contained, relocatable run directory per experiment** (the unit of export and restore):
   ```
   runs/<YYYY-MM-DD>_<name>_s<seed>/
   ├── config.yaml          fully resolved config (all groups)
   ├── metadata.json        git commit, package/torch/CUDA versions, GPU, data checksums,
   │                        tokenizer fingerprint, config hash, sessions log (start/stop per session)
   ├── tokenizer/<name>.model
   ├── checkpoints/
   │   ├── last.pt          full training state (overwritten atomically)
   │   ├── last.prev.pt     previous full state (protection against a corrupt last.pt)
   │   ├── best.pt          weights only, best validation metric
   │   └── avg/step_XXXXXXXX.pt   weights only (fp32), sliding window for checkpoint averaging
   ├── metrics.jsonl        append-only, one JSON object per event (crash-safe)
   └── train.log
   ```
   Paths inside the run directory are **relative**; nothing absolute is persisted.
2. **Full checkpoint contents:** model, optimizer, LR-schedule step, GradScaler, sampler state (`TokenBudgetBatchSampler.state_dict()`: epoch + batches consumed + settings), global step, tokens seen, RNG states (Python, NumPy, Torch CPU, CUDA), config hash, tokenizer fingerprint, best metric. Saved with `torch.save`, loaded with `weights_only=True` (RNG states stored as tensors/lists to allow this).
3. **Atomic writes:** write `*.tmp`, flush + `fsync`, rotate `last.pt → last.prev.pt`, then `os.replace(tmp, last.pt)`. Readers fall back to `last.prev.pt` if `last.pt` fails to load.
4. **Exact resumption contract:** a run interrupted at step k and resumed produces the same subsequent batches, dropout masks and loss values as the uninterrupted run. This is tested bit-exactly on CPU; on GPU it is exact up to non-deterministic kernels.
5. **Session budget:** the trainer takes a wall-clock budget (default 11 h 15 min for a 12 h session) and checkpoints every N steps, at each validation, and when the budget runs out. A SIGTERM handler checkpoints on a best-effort basis.
6. **Restore flow:** attach the previous notebook version's output as an input → `--resume /kaggle/input/<output>/runs/<id>` → copy to `/kaggle/working/runs/<id>` → verify config hash and tokenizer fingerprint → continue.
7. **Storage budget** (mid config, 36.7M parameters): full checkpoint ≈ 0.44 GB (weights + 2 Adam moments), × 2 (last, prev) + 5 × 0.15 GB averaging window + best 0.15 GB ≈ **1.8 GB** per run, well under 20 GB.
8. **Offline-first inputs:** raw data via checksum-verified mirrors (`prepare_data.py --mirror-dir /kaggle/input/<raw-data-dataset>`), the tokenizer reused from the run directory, and the code pinned to a commit (cloned when internet is available, otherwise attached as a dataset), installed with `pip install -e . --no-deps` to keep Kaggle's preinstalled PyTorch.
9. **Per-session preflight:** print versions and GPU name/capability; run `pytest -m "parity or numerical" -q` (seconds on GPU); abort if it fails. This catches PyTorch-version drift (ADR-004) before any GPU-hours are spent.
10. **Configuration for Kaggle paths** goes through overrides (`--override translation.root=/kaggle/working/data`). There is no environment-specific code in `src/`.

## Consequences
**Positive:** sessions are interchangeable; a killed session costs at most N steps; every number is traceable to an exportable directory; local and Kaggle runs share code and configs. **Negative:** checkpoint I/O time (~0.5 GB per save → save every ~15–30 min, not every few steps); the trainer is more complex (Week 5). **Risks:** Kaggle storage or output-format changes → the run directory is plain files, portable to Colab or a local machine.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Save only at epoch end | IWSLT epochs plus Kaggle kills can lose large amounts of work; mid-epoch state is cheap (ADR-009 batches are deterministic) |
| W&B artifacts as the primary store | Needs internet; kept as an optional mirror |
| Weights-only checkpoints | Optimizer and scheduler state are needed for exact resumption |
| Absolute paths in metadata | Break when moving between `/kaggle/input` and `/kaggle/working` |

## Compliance
Week 3: `TokenBudgetBatchSampler.state_dict()/load_state_dict()` + mid-epoch tests; `--mirror-dir` + offline tests. Week 5: an atomic-save test, a kill-and-resume equivalence test, a storage-size test.
