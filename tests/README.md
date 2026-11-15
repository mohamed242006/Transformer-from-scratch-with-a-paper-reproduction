# tests/

Evidence of correctness that does **not** depend on training outcomes.

| Folder | Marker | In CI | Purpose |
|---|---|---|---|
| `policy/` | none | yes | Repository rules: forbidden APIs (ADR-001), package contract |
| `unit/` | none | yes | Behaviour, shapes, invariances, closed-form checks (docs/math.md) |
| `parity/` | `parity` | yes | Numerical equality with PyTorch references (ADR-004) |
| `numerical/` | `numerical` | yes | gradcheck (float64), fp16/bf16 safety, NaN/Inf guards |
| `integration/` | `integration`, `slow` | smoke subset | Overfit, copy task, save/load, end-to-end |

Rules: no network access; no dependence on `data/`; mark slow (> 10 s) and GPU tests.
Contract: [docs/REPOSITORY.md](../docs/REPOSITORY.md#tests).
