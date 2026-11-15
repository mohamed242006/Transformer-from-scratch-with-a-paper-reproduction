# experiments/

Provenance for every reported number. One folder per run:

```
experiments/YYYY-MM-DD_<model>-<data>-<variant>/
├── config.yaml      frozen, fully resolved config
├── metadata.json    git commit + dirty flag, seed, library versions, GPU, data/tokenizer hashes
├── metrics.json     final and best metrics (with sacreBLEU signature)
├── hypotheses.txt   test-set outputs (small text)
└── NOTES.md         hypothesis → result → conclusion
```

**Never committed:** checkpoints/weights, TensorBoard event files, W&B caches, files > 5 MB.
Owner: Research/docs owner. Contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#experiments).
