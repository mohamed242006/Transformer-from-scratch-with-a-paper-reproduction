# IWSLT'14 DE→EN benchmark (Week 9): archived evidence

This folder holds the evidence for every IWSLT'14 number in [docs/results.md](../../docs/results.md) and [ADR-023](../../docs/decisions/ADR-023-iwslt14-benchmark.md). It is a **study archive**: one folder for the study, with one sub-folder per run. Week 9 used 24 runs (pilot, learning-rate selection, Post-LN vs Pre-LN, and the 3-seed benchmark), so one folder per run would be impractical.

## Hypothesis → result → conclusion
- **Hypothesis:** at matched parameters (≈ 36.6M), data, update budget and evaluation, a 6+6 Transformer beats attention RNNs on IWSLT'14 DE→EN (ADR-002, ADR-021).
- **Result:** the Transformer (Pre-LN) scores **32.95 ± 0.05** test BLEU at the protocol defaults, against 31.80 ± 0.19 for BiLSTM + Luong. GRU + Bahdanau's protocol result is 11.51 ± 14.37, because its averaged checkpoint collapsed after training diverged late. With checkpoint and decoding selected on validation, the scores are 33.10 ± 0.07, 32.04 ± 0.10 and 28.88 ± 0.25. Every gap is significant: two-level paired bootstrap, Holm p = 0.002.
- **Conclusion:** the Transformer leads, but by about 1.1 BLEU over the best RNN, far less than in the paper's setting. The paper's Post-LN 6+6 did not train in this recipe (valid BLEU 5.79), so the headline Transformer is a declared Pre-LN deviation. The Post-LN result is bounded, not explained (ADR-023 Amendment 4).

## Contents
| Path | What |
|---|---|
| `study/STUDY.md`, `study/STUDY.json` | Final study report (`configs/study/week9_benchmark.yaml`), produced by `scripts/analyze_study.py` |
| `study/PILOT.md`, `study/PILOT.json`, `study/pilot_curves.csv` | Phase 1 pilot gate report (6k updates, `configs/study/week9_pilot.yaml`) |
| `study/benchmark_curves.csv` | Training and validation curves of all runs |
| `runs/<run>/config.yaml` | Fully resolved config (its hash is in `metadata.json`) |
| `runs/<run>/metadata.json` | Config hash, code-tree hash, tokenizer fingerprint, device, library versions, session times, best update |
| `runs/<run>/metrics.jsonl` | Full training log: loss, NLL, LR, gradient norm, throughput, memory, validation |
| `runs/<run>/eval/*.json` | Every evaluation: sacreBLEU and chrF with signatures, tok-lc BLEU, length ratio, BLEU by source length, decoding speed |
| `runs/iwslt14_{transformer,bilstm_luong,gru_bahdanau}_s{1,2,3}/eval/test.*.hyp.gz` | Test-set hypotheses for the 9 benchmark runs (gzip, one line per sentence) |

Runs by role:

| Role | Runs |
|---|---|
| Phase 1 pilot (6k updates; ADR-023 §8) | `iwslt14_pilot_*` (10 runs) |
| Post-LN vs Pre-LN at full budget (Amendment 2) | `iwslt14_postln_30k_s1` (Post-LN, LR 5e-4), `iwslt14_preln_30k_s1` (Pre-LN, LR 7e-4) |
| Phase 1c learning-rate selection (Amendment 3) | `iwslt14_transformer-pre_ln-lr0.0014_s1`, `iwslt14_bilstm_luong-lr0.0028_s1`, `iwslt14_bilstm_luong-lr0.0112_s1`; the selected runs became seed 1 below |
| Phase 2 benchmark (3 seeds) | `iwslt14_transformer_s{1,2,3}`, `iwslt14_bilstm_luong_s{1,2,3}`, `iwslt14_gru_bahdanau_s{1,2,3}` |

## Environment
- **Hardware:** a single NVIDIA GeForce RTX 5070 Ti (16 GB), fp16 autocast.
- **Software:** Python 3.14.4, PyTorch 2.15.0+cu132, sacreBLEU 2.6.0, sacremoses 0.2.0.
- **Training time:** 8.9 GPU-hours for all 24 runs.
- **Code:** commit `372de77` ("freeze Phase 2 benchmark hyperparameters and study"). The code-tree hash is in each `metadata.json`.
- **Tokenizer fingerprint:** `b0c6c22a…261989`, identical across all runs (ADR-023 §3).

## Re-scoring a number
```bash
python scripts/prepare_data.py --config configs/data/iwslt14.yaml    # rebuilds data/processed/iwslt14 (pinned hashes)
zcat experiments/2026-11-22_iwslt14-week9-benchmark/runs/iwslt14_transformer_s1/eval/test.averaged.beam4.hyp.gz \
  | sacrebleu data/processed/iwslt14/test.en -b -w 2                 # 32.99, as in eval/test.averaged.beam4.json
```
This was checked when the archive was made.

## What is not here
- **Checkpoints and tokenizer models:** never committed (REPOSITORY.md). They are needed to re-decode, and are kept outside the repository on the owner's server. Their home for v1.0 is the private `aiayn-runs` dataset (ADR-020).
- **Validation hypotheses:** used only for selection, and recomputable from checkpoints.
- **`benchmark.log`:** the raw runner log. It contains local absolute paths, and its content is in `metrics.jsonl` and `STUDY.md`.

Regenerating `STUDY.md` with `scripts/analyze_study.py` needs the full run directories, not this archive.
