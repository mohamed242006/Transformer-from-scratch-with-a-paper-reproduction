# ADR-005: SentencePiece is allowed for subword tokenization

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [ADR-001](ADR-001-manual-implementation.md), [ADR-002](ADR-002-iwslt14-before-wmt14.md), [paper_notes.md §6.1, §12 (U12)](../paper_notes.md)

## Context

The paper uses **byte-pair encoding** with a **shared source-target vocabulary of about 37000 tokens** for EN-DE (§5.1), and word-pieces (32k) for EN-FR. A shared vocabulary is **required** by three-way weight tying (§3.4).

ADR-001 requires the *Transformer* to be built by hand. Tokenization is different:
- It is **not part of the paper's contribution**. The paper cites BPE (Sennrich et al., 2016) and word-pieces (Wu et al., 2016) as existing tools.
- A hand-written BPE trainer is a self-contained project (efficient merge counting, Unicode normalization, byte fallback). Getting it wrong would **quietly hurt BLEU** in ways unrelated to the architecture under study.
- Tokenization affects metric comparability (ADR-002). A well-known, deterministic tool makes results reproducible.

At the same time, using a library for tokenization must not blur the line drawn in ADR-001. The rule needs to be explicit.

## Decision

1. Use **SentencePiece** (`sentencepiece` Python package) to train and apply a **joint BPE model** (`model_type=bpe`) on the **training split only** of the concatenated source and target text.
2. **Configuration, fixed in `configs/data/*` and recorded with every experiment:**

   | Setting | Value | Rationale |
   |---|---|---|
   | `model_type` | `bpe` | Matches the paper's BPE |
   | `vocab_size` | ~8k (Multi30k), ~10k (IWSLT14), ~32–37k (WMT14) | Scaled to data size; WMT matches the paper |
   | `character_coverage` | `1.0` | Latin script; keeps German umlauts and ß |
   | Special-token IDs | `pad=0, unk=1, bos=2, eos=3` | Set explicitly. SentencePiece defaults differ (`pad` is disabled by default) and are easy to misconfigure. |
   | `normalization_rule_name` | Recorded explicitly (default `nmt_nfkc`) | Normalization affects BLEU; must be reproducible |
   | Seed / input sentence sampling | Fixed, recorded | Deterministic tokenizer |

3. The tokenizer is wrapped behind a small interface in `src/aiayn/data/` (`encode`, `decode`, `vocab_size`, special IDs). The rest of the code never imports `sentencepiece` directly, so the backend can be swapped (for example for a hand-written BPE extension).
4. The trained `.model`/`.vocab` files are treated as **experiment artifacts**: stored under `data/` (gitignored), checksummed, and referenced by hash in each run's metadata.
5. **Scope boundary (clarifies ADR-001):** libraries are allowed for **data preparation and evaluation infrastructure** (tokenization, dataset download, sacreBLEU), and are **forbidden for the model, the training objective, the optimizer schedule and the decoding algorithms**.
6. A hand-written BPE trainer and encoder is a **listed extension**. If built, it must reproduce SentencePiece's segmentation statistics (vocabulary overlap, average tokens per sentence) on the same data before replacing it.

## Consequences

### Positive
- Reliable, fast, deterministic subword segmentation, and an end-to-end raw-text → IDs → raw-text round trip (SentencePiece handles whitespace with ▁ and needs no separate detokenizer).
- Detokenized output feeds straight into sacreBLEU (the primary protocol, ADR-002).
- Effort stays on the paper's actual contribution.
- The interface keeps the door open for a from-scratch BPE extension.

### Negative / costs
- **Not identical to the paper's tokenizer, nor to fairseq's IWSLT14 pipeline** (Moses tokenization + subword-nmt BPE + lowercasing). Expect BLEU differences of a few tenths when comparing with published IWSLT numbers. This is declared in ADR-002's dual-reporting protocol.
- One more dependency (C++ extension wheels). Wheels exist for Windows, Linux and macOS on current Python versions.
- Training the tokenizer on raw rather than Moses-pretokenized text changes the segmentation of punctuation compared with older pipelines.

### Risks and mitigations
| Risk | Mitigation |
|---|---|
| Tokenizer trained on validation or test data (leakage) | Preparation script takes the training split only; a test asserts the input-file provenance |
| Special-token ID mismatch between tokenizer and model (for example pad = −1) | IDs set explicitly; test asserts `pad=0, unk=1, bos=2, eos=3` and that the embedding `padding_idx` equals `pad_id` |
| Round-trip failures on unusual Unicode | Round-trip test on a sample of the corpus after normalization |
| Different tokenizer between training and evaluation runs | Tokenizer hash stored in the checkpoint metadata; evaluation refuses a mismatch |

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Hand-written BPE now** | High effort, high silent-error risk, outside the paper's contribution. Deferred to an extension. |
| **Moses tokenizer + subword-nmt** (closest to the 2017/fairseq pipelines) | Better literature comparability, but needs Perl or `sacremoses` plus a separate detokenizer, is more fragile on Windows, and requires lowercasing to match the IWSLT references. Our dual-reporting protocol handles comparability instead. |
| **Hugging Face `tokenizers`** | Equally good technically. SentencePiece chosen for its single-file model, built-in detokenization and widespread use in MT. Either would have been acceptable. |
| **Word-level vocabulary** | Out-of-vocabulary problem; not what the paper does |
| **Character or byte level** | Much longer sequences (worse for O(n²) attention); not what the paper does |

## Compliance
- `configs/data/*.yaml` must contain every key in the table in Decision 2. The config loader validates this (Week 2).
- Tests: special-token IDs, round trip, no train/test overlap in tokenizer training input, tokenizer-hash check at evaluation time.
- `sentencepiece` may be imported only inside `src/aiayn/data/` (policy test extended in Week 2).
