# scripts/

Thin CLI entry points: argument parsing + config loading + calls into `aiayn`. Any logic longer than ~30 lines belongs in `src/`.

Planned: `prepare_data.py`, `train_tokenizer.py` (Week 2) · `train.py` (Week 5) · `evaluate.py`, `translate.py`, `average_checkpoints.py` (Week 6) · `benchmark.py` (Week 9).

Every script defines `main()` and an `if __name__ == "__main__":` guard (required on Windows, where DataLoader workers are spawned).
Contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#scripts).
