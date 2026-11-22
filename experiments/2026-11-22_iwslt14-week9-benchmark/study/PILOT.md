# Pilot report: `week9_pilot`

Gates and thresholds: ADR-023 §8. Every number is read from the run files.

**Status:** 10/10 runs completed.

## 1. Validation curves (teacher-forced NLL)

| Update | transformer:default | transformer:lr0.0005 | transformer:lr0.001 | transformer:pre_ln | gru_bahdanau:default | gru_bahdanau:lr0.0014 | gru_bahdanau:lr0.0028 | bilstm_luong:default | bilstm_luong:lr0.0014 | bilstm_luong:lr0.0028 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 5.521 | 5.657 | 5.445 | 4.728 | 4.894 | 4.371 | 3.609 | 5.237 | 4.527 | 3.881 |
| 2000 | 4.647 | 4.771 | 4.612 | 3.419 | 3.490 | 2.820 | 2.488 | 3.800 | 3.027 | 2.607 |
| 3000 | 4.161 | 4.250 | 4.174 | 2.678 | 2.719 | 2.519 | 2.306 | 2.882 | 2.482 | 2.234 |
| 4000 | 3.912 | 3.948 | 3.978 | 2.303 | 2.381 | 2.203 | 2.235 | 2.501 | 2.249 | 2.092 |
| 5000 | 3.694 | 3.687 | 3.786 | 2.100 | 2.202 | 2.089 | 2.127 | 2.260 | 2.052 | 1.956 |
| 6000 | 3.537 | 3.511 | 3.660 | 1.933 | 2.123 | 2.028 | 2.076 | 2.136 | 1.957 | 1.874 |

## 2. Throughput, memory and stability (T4 training logs)

| Run | Peak LR | Batch (tokens x accum) | Target tok/s | Peak MiB | Skipped fp16 | Max grad norm | Final train NLL | Best valid NLL | Valid BLEU | Min per 1k updates | Phase 2 run (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `iwslt14_pilot_transformer_s1` | 0.0007 | 8192 x 1 | 112,542 | 4558 | 2 | 2.95 | 3.627 | 3.537 | 3.82 | 1.1 | 0.5 |
| `iwslt14_pilot_transformer-lr0.0005_s1` | 0.0005 | 8192 x 1 | 112,440 | 4558 | 2 | 2.86 | 3.621 | 3.511 | 4.32 | 1.1 | 0.5 |
| `iwslt14_pilot_transformer-lr0.001_s1` | 0.001 | 8192 x 1 | 112,499 | 4558 | 2 | 2.53 | 3.756 | 3.660 | 2.95 | 1.1 | 0.5 |
| `iwslt14_pilot_transformer-pre_ln_s1` | - | 8192 x 1 | 110,847 | 4621 | 1 | 1.92 | 1.976 | 1.933 | 27.05 | 1.1 | 0.5 |
| `iwslt14_pilot_gru_bahdanau_s1` | 0.0007 | 4096 x 2 | 70,304 | 8622 | 1 | 3.04 | 2.184 | 2.123 | 26.62 | 1.8 | 0.9 |
| `iwslt14_pilot_gru_bahdanau-lr0.0014_s1` | 0.0014 | 4096 x 2 | 70,494 | 8622 | 1 | 3.00 | 2.049 | 2.028 | 28.52 | 1.8 | 0.9 |
| `iwslt14_pilot_gru_bahdanau-lr0.0028_s1` | 0.0028 | 4096 x 2 | 70,506 | 8622 | 1 | 2.05 | 2.225 | 2.076 | 28.62 | 1.8 | 0.9 |
| `iwslt14_pilot_bilstm_luong_s1` | 0.0007 | 8192 x 1 | 187,799 | 2143 | 1 | 1.46 | 2.246 | 2.136 | 26.09 | 0.6 | 0.3 |
| `iwslt14_pilot_bilstm_luong-lr0.0014_s1` | 0.0014 | 8192 x 1 | 187,898 | 2143 | 1 | 1.55 | 1.993 | 1.957 | 28.75 | 0.6 | 0.3 |
| `iwslt14_pilot_bilstm_luong-lr0.0028_s1` | 0.0028 | 8192 x 1 | 187,875 | 2143 | 1 | 1.35 | 1.872 | 1.874 | 30.45 | 0.6 | 0.3 |

## 3. Learning-rate selection (validation BLEU, NLL breaks ties)

- **transformer**: selected `lr0.0005` (peak LR 0.0005) ⚠ train.schedule.peak_lr at grid minimum: add one pilot run beyond the edge before Phase 2. Candidates: 0.0007: BLEU 3.82 / NLL 3.537, 0.0005: BLEU 4.32 / NLL 3.511, 0.001: BLEU 2.95 / NLL 3.660.
- **gru_bahdanau**: selected `lr0.0028` (peak LR 0.0028) ⚠ train.schedule.peak_lr at grid maximum: add one pilot run beyond the edge before Phase 2. Candidates: 0.0007: BLEU 26.62 / NLL 2.123, 0.0014: BLEU 28.52 / NLL 2.028, 0.0028: BLEU 28.62 / NLL 2.076.
- **bilstm_luong**: selected `lr0.0028` (peak LR 0.0028) ⚠ train.schedule.peak_lr at grid maximum: add one pilot run beyond the edge before Phase 2. Candidates: 0.0007: BLEU 26.09 / NLL 2.136, 0.0014: BLEU 28.75 / NLL 1.957, 0.0028: BLEU 30.45 / NLL 1.874.

## 4. Gates

| Gate | Status | Evidence |
|---|---|---|
| G1 Post-LN trains | PASS | completed=True, best at last evaluation=True, decreasing over last 3=True (3.912, 3.694, 3.537), skipped 2/6000 |
| G2 Post-LN not worse than Pre-LN | **FAIL** | best valid NLL Post 3.537 vs Pre 1.933 (limit +0.05); valid BLEU Post 3.82 vs Pre 27.05 (limit -0.5) |
| G3 learning rates bracketed | **FAIL** | see section 3 |
| G4 Phase 2 run fits a session | PASS | transformer 0.5 h/run, gru_bahdanau 0.9 h/run, bilstm_luong 0.3 h/run; total 5.2 h for 3 seeds (~0.2 weekly quotas of 30 h) |

**Post-LN decision:** **NO-GO**: Phase 2 headline Transformer switches to Pre-LN (ADR-023 amendment).

## 5. Phase 2 recommendation

- Transformer: `model.norm_position=pre`.
- transformer: `train.schedule.peak_lr=0.0005`. **First run one more pilot point beyond the grid edge**; the optimum is not bracketed.
- gru_bahdanau: `train.schedule.peak_lr=0.0028`. **First run one more pilot point beyond the grid edge**; the optimum is not bracketed.
- bilstm_luong: `train.schedule.peak_lr=0.0028`. **First run one more pilot point beyond the grid edge**; the optimum is not bracketed.
- 3 seeds per model, 30k updates, valid-selected checkpoint and decoding (ADR-023 §7); projected 5.2 T4-hours in total.
