# ADR-010: Compute platform: local CPU for development, free cloud GPUs for training

- **Status:** Accepted (approved by project owner 2026-11-15; Kaggle chosen as primary platform)
- **Date:** 2026-11-15
- **Deciders:** Lead engineer, project owner
- **Related:** [ADR-002](ADR-002-iwslt14-before-wmt14.md), spec §5, §6.2, §13

## Context
Hardware audit of the development machine (2026-11-15): Intel Core i5-1240P (12 cores / 16 threads), **Intel Iris Xe integrated graphics (no CUDA GPU)**, **7.6 GB RAM**, ~26 GB free disk, Windows 11.

Rough training cost (6·P·tokens, math.md §10.3):

| Workload | Params | Tokens per epoch (src + tgt) | FLOPs per epoch | Laptop CPU (~50 GFLOP/s effective) | Free cloud T4/P100 (~5–10 TFLOP/s effective) |
|---|---|---|---|---|---|
| Copy task (tiny model) | ~1–2M | ~0.5M | ~5·10¹² | ~2 min | seconds |
| Multi30k, beginner config | 7.6M | ~0.8M | ~4·10¹³ | ~15 min | ~5–10 s of compute (I/O-bound in practice) |
| IWSLT14, mid config | 36.7M | ~8M | ~1.8·10¹⁵ | **~10 h** | ~3–6 min |

These are order-of-magnitude estimates. Hand-written attention without fused kernels will run below these utilizations.

Free GPU tiers (Kaggle ≈ 30 GPU-h/week; Colab, variable) impose **session limits of about 12 h**.

## Decision
1. **Local CPU:** all development, unit/integration tests, the copy task, and short Multi30k smoke runs.
2. **Free cloud GPU (Kaggle primary, Colab fallback):** Multi30k training runs and **all ablations** (ADR-002's declared fallback becomes the plan), plus **IWSLT14 headline runs only** (Transformer + strongest RNN baseline × 3 seeds).
3. **WMT14 is removed from the roadmap.** It is reconsidered only with a paid budget.
4. **Training must be resumable** (Week 5 requirement): checkpoints include model, optimizer, scheduler step, sampler epoch, and Python/NumPy/Torch RNG states. A resumed run must reproduce the uninterrupted run's loss curve (integration test).
5. Notebooks or scripts for the cloud must install the package from the repository and read the same `configs/`. No logic is duplicated in notebooks.
6. Intel XPU acceleration for the Iris Xe GPU is **not** pursued (limited support for integrated graphics; not worth the risk).

## Consequences
**Positive:** the plan is feasible at zero cost; the ablation table gets ≥ 3 seeds per row on Multi30k. On Kaggle P100 hardware, results run on the same GPU model as the paper, which makes the MFU comparison direct. **Negative:** ablation conclusions come from small data (stated as such); dependence on free-tier quotas and availability; IWSLT has only headline rows. **Risks:** quota changes → Colab fallback and smaller seeds-per-row; session preemption → resumability (item 4).

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Train everything locally on CPU | IWSLT ~10 h/epoch: infeasible |
| Paid cloud GPU for everything | Unnecessary cost for this scale; kept as an option for WMT14 |
| Shrink the IWSLT model to fit CPU | Destroys comparability with published IWSLT references |
| Intel XPU on Iris Xe | Immature for integrated GPUs; debugging risk outweighs gain |
