# ADR-004: Numerical parity testing against PyTorch reference implementations is mandatory

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [ADR-001](ADR-001-manual-implementation.md), [ADR-003](ADR-003-post-ln-default.md), [math.md §0, §1.3, §8](../math.md), spec §10

## Context

ADR-001 requires hand-written components, and hand-written attention code has a well-known failure mode: **tensors with the right shapes and the wrong values**. Examples:
- Reshaping `[B, T, d] → [B, h, T, d_k]` without the intermediate transpose scrambles positions across heads. Every shape check passes.
- A softmax over the wrong axis still returns a valid probability distribution.
- A missing √d_k scale still trains, just worse.
- An extra LayerNorm, or a missing dropout location, changes results by a fraction of a BLEU point.

**BLEU cannot detect these bugs.** A slightly wrong Transformer still learns to translate, so a mistake would show up only as a vague "our BLEU is 1.5 points low". We need a test that is independent of training outcomes.

PyTorch's own modules implement the same mathematics and can serve as **reference implementations**: `nn.LayerNorm`, `F.scaled_dot_product_attention`, `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer` and `F.cross_entropy(label_smoothing=…)`.

Known differences between our implementation and these references, which the tests must bridge:

| Difference | Our convention | PyTorch reference | Bridge |
|---|---|---|---|
| Boolean mask meaning | **True = may attend** | `nn.MultiheadAttention.key_padding_mask` and `attn_mask`: **True = masked out**. `F.scaled_dot_product_attention` boolean `attn_mask`: True = takes part. | Explicit conversion helpers in `tests/parity/` |
| Q/K/V projection storage | Three `nn.Linear(d, d)` | One packed `in_proj_weight` of shape `[3d, d]` + `in_proj_bias` | Concatenate our weights (Q, K, V order) when copying |
| Stack-level final norm | None in Post-LN (ADR-003) | `nn.Transformer` adds encoder and decoder final LayerNorms by default | Build references from `nn.TransformerEncoder(norm=None)` / `nn.TransformerDecoder(norm=None)`, not `nn.Transformer` |
| Inference fast path | n/a | Encoder layers may take a fused "fast path" in eval mode (different numerics, nested tensors) | Compare in train mode with dropout = 0, or with inputs that disable the fast path |
| Batch dimension | `[B, T, d]` | `batch_first=False` by default | Always construct references with `batch_first=True` |
| Embeddings, PE, generator | Part of our model | Not part of `nn.Transformer*` | Parity covers the layer stacks. Embeddings, PE and generator are tested against closed forms (math.md §4, §5). |

## Decision

1. **Every component that has a PyTorch counterpart must have a parity test before it is used in any training run.** No experiment result is recorded for a model whose parity tests fail.
2. **Parity test inventory** (tests live in `tests/parity/`, marked `@pytest.mark.parity`):

   | Our component | Reference implementation | Weight copy |
   |---|---|---|
   | LayerNorm | `nn.LayerNorm(d, eps=1e-5)` | γ, β |
   | Scaled dot-product attention | `F.scaled_dot_product_attention` (and a naive float64 einsum version) | none |
   | Multi-head attention (self and cross, with masks) | `nn.MultiheadAttention(batch_first=True)` | Pack Q, K, V into `in_proj_*`; copy `out_proj` |
   | Encoder layer (Post- and Pre-LN) | `nn.TransformerEncoderLayer(norm_first=False/True, activation="relu", batch_first=True)` | Layer-by-layer mapping |
   | Decoder layer (Post- and Pre-LN) | `nn.TransformerDecoderLayer(...)` | Layer-by-layer mapping |
   | Encoder / decoder stacks | `nn.TransformerEncoder(norm=None)` / `nn.TransformerDecoder(norm=None)` (Pre-LN: `norm=LayerNorm`) | Per layer |
   | Label-smoothed loss | `F.cross_entropy(label_smoothing=ε, ignore_index=pad)` | none |

3. **Tolerance tiers** (asserted with `torch.testing.assert_close`):

   | Setting | `atol` | `rtol` | Notes |
   |---|---|---|---|
   | Single module, float64 | 1e-10 | 1e-8 | Primary correctness check: removes floating-point noise |
   | Single module, float32 | 1e-6 | 1e-5 | |
   | Full stack (N = 6), float32 | 1e-5 | 1e-4 | Errors accumulate through layers |
   | Gradients (`.grad` of inputs and all parameters) | as above per dtype | | Backward pass parity too, not only forward |

4. **Test discipline:** dropout = 0 (or `eval()` with the fast path disabled); fixed seeds; deterministic algorithms where available; several shapes per test, including `B = 1`, `T = 1`, a non-square `T_q ≠ T_k` for cross-attention, and padded batches with ragged lengths.
5. **Parity tests are complemented, not replaced, by** invariance tests (padding invariance, causality), closed-form tests (PE rotation, Noam values, parameter counts) and `torch.autograd.gradcheck` in float64.
6. Reference implementations are imported **only** under `tests/`, enforced by the ADR-001 policy test.

## Consequences

### Positive
- **Correctness is established independently of training,** so BLEU gaps can be attributed to data, tuning or protocol, not hidden bugs.
- Weight-mapping adapters force a precise understanding of PyTorch's internal layouts, which is a valuable skill in its own right.
- A concrete CV and README claim: "verified to within 1e-6 (fp32) of PyTorch's reference implementation, forward and backward".
- Parity tests protect future refactors (KV cache, optimized backends).

### Negative / costs
- Writing the adapters takes about 1–2 days in Weeks 3–4.
- Parity ties our layout choices to PyTorch's conventions where convenient. That is acceptable: the adapters isolate it.
- PyTorch version changes (for example fast-path behaviour) can break the tests without our code changing. Mitigated by pinning the PyTorch version range in `pyproject.toml` and running parity tests in CI.

### Risks and mitigations
| Risk | Mitigation |
|---|---|
| Both implementations share a *conceptual* mistake (for example the same wrong mask meaning in our adapter and our code) | Independent invariance tests (padding and causal perturbation) do not depend on any reference |
| Tolerances loosened until tests pass | Tolerances are fixed in this ADR. Changing them needs a new ADR with justification. |
| A reference module behaves differently across devices | Parity runs on CPU in CI. Optional GPU runs are marked `gpu`. |

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Rely on shape tests + "it trains" | Cannot detect scrambled heads, wrong softmax axis or a missing scale; BLEU is too blunt |
| Compare against the Annotated Transformer or another repository | It is Pre-LN (ADR-003), not maintained as a reference implementation, and adds a dependency on third-party correctness |
| Compare against Hugging Face models (for example Marian) | Different architectural details (normalization, biases, scaling); weight mapping is fragile |
| Gradcheck only | Verifies gradients match *our own* forward pass, not that the forward pass matches the paper |

## Compliance
- CI runs `pytest -m parity` on every push once the tests exist (Week 3+).
- Pull-request checklist: "New component? Parity test added, or justified as having no reference implementation."
- `docs/experiment_log.md` entries reference the commit at which parity passed.
