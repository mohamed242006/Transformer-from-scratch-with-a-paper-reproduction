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

**Studies** archive many runs in one folder, `experiments/YYYY-MM-DD_<data>-<study>/`, with `study/` (reports, curves), `runs/<run>/` (config, metadata, metrics, evaluations) and a `NOTES.md`. Example: [2026-11-22_iwslt14-week9-benchmark](2026-11-22_iwslt14-week9-benchmark/NOTES.md).

**Never committed:** checkpoints/weights, TensorBoard event files, W&B caches, files > 5 MB.
Owner: Research/docs owner. Contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#experiments).
