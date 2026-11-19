# data/ (gitignored)

Local cache for datasets and tokenizers. **Nothing here except this README is committed**: dataset licences (e.g. IWSLT/TED, CC BY-NC-ND) forbid redistribution, and the files are large.

```
data/
├── raw/          downloaded archives, verified against checksums in configs/data/
├── processed/    cleaned, split, length-filtered text
└── tokenizers/   SentencePiece .model / .vocab (hash recorded in each run's metadata)
```

Everything here is reproducible with `scripts/prepare_data.py` and `scripts/train_tokenizer.py` (Week 2).
IWSLT'14 (`configs/data/iwslt14.yaml`, ADR-023) is built from one pinned archive; `processed/iwslt14/MANIFEST.json` records its counts and file hashes.
Owner: Data owner. Contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#data-gitignored).
