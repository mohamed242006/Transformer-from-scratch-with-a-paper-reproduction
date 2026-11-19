# ADR-023: IWSLT'14 DE→EN benchmark: data, evaluation protocols, incremental decoding, study rules and pilot gate

- **Status:** Accepted (Phase 0 implemented 2026-11-19; Phases 1–2 pending approval)
- **Date:** 2026-11-19
- **Deciders:** Lead engineer, project owner
- **Related:** [ADR-002](ADR-002-iwslt14-before-wmt14.md), [ADR-005](ADR-005-sentencepiece-tokenization.md), [ADR-006](ADR-006-evaluation-protocol.md), [ADR-019](ADR-019-decoding-and-evaluation.md), [ADR-021](ADR-021-rnn-baselines-and-fair-comparison.md), [ADR-022](ADR-022-multi-seed-study-and-statistics.md), [results.md](../results.md)

## Context
Week 8 established the Multi30k findings (results.md). ADR-002 makes IWSLT'14 DE→EN the headline benchmark: 160k TED sentence pairs and the standard small-scale Transformer benchmark. Before spending GPU time, Week 8 identified six things to fix:
1. No pinned IWSLT source or split definition.
2. No metric comparable with the literature, which reports *tokenized, lowercased* BLEU.
3. Decoding cost: prefix recomputation is quadratic in output length. Prohibitive for 6,750 longer test sentences and the Bahdanau loop.
4. Validation-NLL selection produced a coin flip (0.0002 nats).
5. The checkpoint-averaging window hurt the RNNs.
6. A 6+6 Post-LN Transformer failed under a short schedule (`layers6`).

The owner chose protocol **A + B**: raw cased text with 10k SentencePiece and sacreBLEU as the primary metric, *plus* fairseq-style tokenized lowercased BLEU.

## Decision

### 1. Source (pinned)
| | |
|---|---|
| URL | `https://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz` (fairseq's public copy of the WIT3 2014-01 release; the original WIT3 links are unreliable) |
| SHA-256 | `2c0e4d19e99384ffc7a395d6639297f98fd86a424b02a2ac1f52cd1ff85cf275` |
| Size | 19,982,877 bytes, last modified 2021-01-07 |
| Members used | `train.tags.de-en.{de,en}`, `IWSLT14.TED.{dev2010,tst2010,tst2011,tst2012}.de-en.{de,en}.xml`, `IWSLT14.TEDX.dev2012.de-en.{de,en}.xml` |
| Licence | TED talks, CC BY-NC-ND: never committed or redistributed (`data/` is gitignored; the `aiayn-runs` dataset is private) |

The archive is read member by member (`tarfile.extractfile`), so nothing is extracted to disk and archive paths cannot escape.

### 2. Splits: fairseq's `prepare-iwslt14.sh` membership, raw text (`aiayn/data/iwslt.py`)
- **Train source:** drop lines containing `<url>`, `<talkid>`, `<keywords>`; strip `<title>`/`<description>` tags.
- **Cleaning:** `clean-corpus-n.perl -ratio 1.5 … 1 175`, decided on Moses-tokenized text (escaping on), exactly as fairseq does. Kept pairs are written **raw** (cased, untokenized).
- **Valid** = every 23rd kept pair; **train** = the rest.
- **Test** = dev2010 + TEDX.dev2012 + tst2010 + tst2011 + tst2012, in that order.

| Split | This pipeline (sacremoses 0.2.0) | fairseq (Moses `tokenizer.perl`) |
|---|---|---|
| train | 160,215 | 160,239 |
| valid | 7,282 | 7,283 |
| test | **6,750** | **6,750** |

- **Test is identical** to fairseq's: the cleaning step doesn't touch it.
- **Train + valid keep 25 fewer pairs** (0.015%). Likely cause: the Python port of the Moses tokenizer differs from `tokenizer.perl` on a few lines near the length and ratio thresholds.
  - Because valid is "every 23rd *kept* pair", this also shifts which pairs land in valid.
  - Valid is used only for selection, never reported as a result.
  - Comparing against the original Perl scripts was not possible in the development environment; see *Open items*.
- **Guarding against drift:**
  - `sacremoses` is pinned (`==0.2.0`), since its tokenization decides membership.
  - The counts are pinned in `configs/data/iwslt14.yaml`; a mismatch raises.
  - `processed/iwslt14/MANIFEST.json` records the source hash, the counts and every output file's SHA-256. Preparation is idempotent: about 60 s once, then about 0.4 s.

Output hashes (raw text, LF-terminated):

| File | SHA-256 |
|---|---|
| train.de | `dff9f126de19e0beac8a3391f0952bfa951e7aad246af153224ad5299942f444` |
| train.en | `5df09b737c3c3df646467efb5ccf69c68ddcf6a39e060f4c43904efa323b3326` |
| valid.de | `b0f63aa652252b3733f46950d0de516bc225141ec4d03a5f6de8b87154ff7867` |
| valid.en | `f6df24b6a303067762d296b9b6b513c0a812cb3010cda10c66cc8715fc1a35f4` |
| test.de | `626d1dd3a37e043028cdf333ecadd676ba832c9773b7d896793cc3c9160bb8a1` |
| test.en | `ddb498b75691c14b9680e55e12fc5b84506b5dbff768191bacc7034cfa526d42` |

### 3. Protocol A (primary): raw cased text, SentencePiece, sacreBLEU
- Joint SentencePiece BPE, **10k** vocabulary, on the raw training text with the ADR-005 settings (all sentences, 1 thread).
- The tokenizer is deterministic. Its fingerprint, `b0c6c22a45b05ef17842a299f3e41bc7ccb971d3bfd90dab8a3858872b261989`, is pinned in the validation notebook.
- **Primary metric:** sacreBLEU, cased, detokenized, `tok:13a`, against the raw test references. Also reported: chrF, sacreBLEU lowercase, length ratio.
- **Direction:** DE→EN, as in the literature.

### 4. Protocol B (comparability): fairseq-style tokenized lowercased BLEU (`bleu_fairseq`)
- **Normalization**, applied to hypotheses and references alike (`aiayn.evaluation.metrics.fairseq_normalize`), in the same order as `prepare-iwslt14.sh` processes the test references:
  1. typographic apostrophe U+2019 → `'`;
  2. Moses tokenization (`sacremoses`, escaping on);
  3. lowercase.
- **Scoring:** corpus BLEU with `tokenize=none, smooth=none`, which equals `multi-bleu.perl` and `fairseq-score` at corpus level. The signature is stored with every result.
- **Comparability caveat, stated wherever the number appears:** fairseq-reported systems (Transformer ≈ 34–35) are *trained* on Moses-tokenized lowercased text. Our models train on raw cased text, a harder task, and are lowercased only for scoring. Applying the apostrophe normalization to hypotheses too removes one systematic mismatch the fairseq pipeline keeps. Protocol B therefore places our numbers on the literature scale; it is not a reproduction of fairseq's pipeline.
- `bleu_fairseq` is computed for every translation evaluation, Multi30k included, and shown in the study tables.

### 5. Schema evolution without hash changes
New config fields marked `OMIT_IF_DEFAULT` (`aiayn.utils.config`) are left out of serialization while at their default. The new `translation.archive` source therefore leaves every existing config's serialized form and `config_hash` unchanged. Verified: `multi30k_tiny` = `87b80534…` before and after. Without this, completed Week 6–8 runs would fail their hash check when re-evaluated through `train.py --resume`.

### 6. Incremental decoding
Every `EncoderDecoder` now implements `init_state` / `decode_step` / `reorder_state`:
- **Transformer:** each decoder layer caches its self-attention keys and values; cross-attention keys and values over the encoder memory are projected once (`MultiHeadAttention.project_kv` / `attend`). The step reuses the same sub-layer blocks as the full forward pass.
- **RNNs:** carried recurrent state. Bahdanau's time step is one function shared by training and decoding.
- **Beam search** reorders the state with the reselected rows.

Greedy and beam search use the cache when available; `incremental=False` keeps the prefix path.

- **Exactness (tests):** every step equals full-prefix decoding to 1e-10 in fp64 for Post-LN and Pre-LN Transformers and all RNN variants. Greedy, beam 4 and beam 5 (α 1.0) produce identical hypotheses with and without the cache.
- **Speed** (CPU, 16 sentences, beam 4, 40 tokens):

  | Model | Prefix | Incremental | Speed-up |
  |---|---|---|---|
  | transformer_tiny | 4.46 s | 0.88 s | 5.1× |
  | gru_bahdanau | 5.77 s | 0.61 s | 9.5× |

  The speed-up grows with output length.

### 7. Study rules adopted from Week 8 (`aiayn.study`)
1. **`training_selection: valid_bleu`** (default): select by mean validation BLEU at protocol decoding, with mean best-validation NLL as the tie-break. Sensitivity runs and main runs get a validation evaluation.
2. **`select_checkpoint: true`** (default): best vs averaged checkpoint chosen per model on validation, before the decoding grid. The grid runs on the chosen checkpoint, and the test evaluation uses both choices.
3. **Variants per model family:** `{overrides: [...], models: [...]}`, e.g. separate learning-rate grids for the Transformer and the RNNs. Ablation and sensitivity groups are optional.
4. **Grid-edge flags** for training choices (an override value equal to its grid's min or max, the model default included) and decoding choices.
5. **Training-health table:**
   - columns: best-update position, final train NLL, best validation NLL, max gradient norm;
   - flags: *early best* (< 60% of training) and *under-fitting* (final train NLL > model default's + 0.3).
6. **Outlier seeds:** test BLEU more than 3 below the configuration's median.
7. **Fairseq column:** a tok-lc BLEU column in every comparison table.

`configs/study/week8.yaml` pins the Week 8 rules (`valid_nll`, `select_checkpoint: false`), so its report stays reproducible. The averaging window for IWSLT is 5 × 1,000 of 30,000 updates (17%).

### 8. Phase 1 pilot and go/no-go gate (`configs/study/week9_pilot.yaml`)
**10 runs**, seed 1, 6,000 updates each (`iwslt_pilot`: the `iwslt_t4` recipe with warmup 4,000, peak 7e-4, 8,192 tokens per batch, cut short):

| Model | Variants (peak learning rate) |
|---|---|
| Transformer 6+6 Post-LN, d = 512 (36.66M) | 5e-4, **7e-4** (default), 1e-3; plus a **Pre-LN** arm at 7e-4 |
| BiGRU + Bahdanau, 1 × 1104 (36.57M) | **7e-4**, 1.4e-3, 2.8e-3 |
| BiLSTM + Luong, 2 × 720 (36.61M) | **7e-4**, 1.4e-3, 2.8e-3 |

These grids bracket each default from both sides; Week 8's optima all sat at the upper edge.

**Gate (checked in `STUDY.md` §4, §7 and §9):**
- **G1. Post-LN trains.** The default Transformer completes; its best validation is at the last evaluation (still improving at 6k updates); no *under-fitting* or *early best* flag; no non-finite steps.
- **G2. Post-LN is not worse than Pre-LN.** Best validation NLL(Post) ≤ NLL(Pre) + 0.05, and validation BLEU(Post) ≥ BLEU(Pre) − 0.5.
- **If G1 or G2 fails,** the Phase 2 headline Transformer is Pre-LN. This is recorded as an amendment here, and the Post-LN result is reported as a finding (it bears on Week 8's `layers6`).
- **G3. Learning rate.** Per model, the variant with the best validation BLEU sets Phase 2's `peak_lr`. If the choice is flagged *at grid edge*, one more pilot run beyond the edge precedes Phase 2.
- **G4. Budget.** Measured tokens/s from `kaggle_validate` (all three IWSLT models) and from the pilots fix the Phase 2 plan. RNN throughput at 36M parameters has not been measured yet.

**Gate evaluation is automated:** `aiayn.study.pilot` and `scripts/pilot_report.py` produce `PILOT.md` and `PILOT.json` (gates, evidence, curves, throughput, memory, selections, recommendation); cell 5b of the study notebook runs them.

**Phase 2** (after approval): `configs/experiment/iwslt14_{transformer,gru_bahdanau,bilstm_luong}.yaml` (30k updates). `peak_lr` is set from G3, 3 seeds each, with the §7 study rules.

## Consequences
**Positive:**
- **Literature comparison:** the test set is identical to the literature's, and both protocols are reported.
- **Decoding:** IWSLT-scale decoding and validation-based selection become affordable.
- **Week 8 lessons are in the code:** selection rules, edge flags, health and outlier reporting.
- **Gated launch:** the deep Post-LN risk is checked before the main budget is spent.
- **Back-compatible:** existing runs, configs and hashes are unchanged.

**Negative:**
- **Membership:** train/valid differ from fairseq's by 25 pairs.
- **New dependency:** `sacremoses` is pinned exactly.
- **Comparability:** Protocol B numbers are comparable to the literature, not identical to it.
- **Pilot budget:** about 10 short runs. RNN cost is the main unknown until G4.

**Open items:**
- **Exact fairseq membership.** Running Moses' `tokenizer.perl` + `clean-corpus-n.perl` (mosesdecoder `08e7820`) on the same tag-filtered text would show whether the 25 pairs come from the tokenizer port. If exact fairseq membership is wanted, the dropped-line indices can then be pinned as a small resource file.
- **No LSTM floor on IWSLT.** The no-attention LSTM is omitted, since Week 8 settled it.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Train on fairseq's lowercased Moses-tokenized text | Owner chose raw cased text (protocol A); protocol B covers comparability |
| HF datasets / torchtext copies of IWSLT14 | Unpinned, different splits, or 2016/2017 releases |
| Shell out to Moses Perl scripts during preparation | Extra downloads and a Perl dependency on every platform; drift is caught by pinned counts and hashes |
| Keep prefix decoding | 5–10× slower at 40 tokens, worse for longer sentences; blocks validation-based selection |
| Add `architecture`/`archive` keys to every config | Changes every existing config hash (§5) |

## Compliance
- `tests/unit/test_iwslt.py`:
  - the recipe (tag filtering, segments, cleaning, every-23rd split, test order);
  - misalignment and count errors;
  - idempotent manifest; schema evolution; config validation.
- `tests/unit/test_metrics.py`: fairseq normalization and tok-lc BLEU.
- `tests/unit/test_incremental_decoding.py`: step equivalence, row reordering, and identical search output for every model type.
- `tests/unit/test_study_plan.py` and `tests/integration/test_study_runner.py`:
  - the selection rules (valid BLEU, NLL tie-break, checkpoint, per-model variants, edge flags);
  - the ADR-023 rules end to end.
- `tests/policy/test_layering.py`: `sacremoses` is allowed in `data` and `evaluation` only.
- `tests/unit/test_pilot.py`: gate verdicts in GO, NO-GO and pending cases; the G2 margin; edge flags; Phase 2 time projection; validation-report parsing; script exit codes.
