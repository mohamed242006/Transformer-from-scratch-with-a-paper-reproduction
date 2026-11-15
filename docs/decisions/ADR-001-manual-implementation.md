# ADR-001: Implement the Transformer manually instead of using `nn.Transformer`

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [ADR-004](ADR-004-parity-testing.md), spec §2.2, [REPOSITORY.md](../REPOSITORY.md)

## Context

PyTorch ships production-quality building blocks: `nn.Transformer`, `nn.TransformerEncoderLayer`, `nn.MultiheadAttention`, `nn.LayerNorm` and `F.scaled_dot_product_attention` (which dispatches to FlashAttention or memory-efficient kernels). Using them would give a working model in about 100 lines.

The project has two goals that conflict with this:

1. **Demonstrate understanding.** The deliverable is evidence that the author can derive, implement and debug every component: masking, head splitting, normalization, scaling and initialization. A call to `nn.Transformer` shows none of that.
2. **Experiment with internals.** Planned ablations and analyses need hooks that the fused modules hide or make awkward: Post-LN vs Pre-LN on the same code path, per-head attention weights and entropy, removing the √d_k scale, learned vs sinusoidal PE, and later KV caching and RoPE.

The usual counter-argument is that hand-written code is more likely to be wrong. That risk is real and is addressed in ADR-004.

## Decision

1. Every component in the paper's model definition is implemented **by hand** under `src/aiayn/`: embeddings with scaling and tying, positional encoding, scaled dot-product attention, multi-head attention, LayerNorm, the residual/dropout sublayer wrapper, FFN, encoder and decoder layers and stacks, the full model, masks, label-smoothed loss, the Noam schedule, greedy and beam decoding, and checkpoint averaging.
2. These PyTorch APIs are **forbidden anywhere under `src/`**:
   - `torch.nn.Transformer`, `TransformerEncoder`, `TransformerDecoder`, `TransformerEncoderLayer`, `TransformerDecoderLayer`
   - `torch.nn.MultiheadAttention`
   - `torch.nn.functional.scaled_dot_product_attention`, `torch.nn.functional.multi_head_attention_forward`
   - `torch.nn.LayerNorm`, `torch.nn.functional.layer_norm`
   - Any import of Hugging Face `transformers`
3. These APIs are **allowed in `src/`**: tensor operations, autograd, `nn.Module`, `nn.Parameter`, `nn.Linear`, `nn.Embedding`, `nn.Dropout`, `nn.ModuleList`, `torch.optim.Adam`, and `nn.LSTM`/`nn.GRU` (baselines only; they are not the subject of the reproduction).
4. The forbidden APIs are **allowed, and encouraged, in `tests/`** as reference implementations (ADR-004).
5. Optimized kernels (for example SDPA/FlashAttention) may come back later only as an **opt-in, separately benchmarked backend** under a new ADR. The hand-written path stays the default and the reference.

## Consequences

### Positive
- Every line of the model is explainable in an interview, and the code doubles as teaching material.
- Ablations and instrumentation (attention maps, per-head statistics, normalization placement) are first-class configuration options rather than monkey-patches.
- The pipeline from math.md derivation to code to test is visible to reviewers.

### Negative / costs
- **Slower training** than fused kernels (no FlashAttention, no fused LayerNorm). This is acceptable at IWSLT scale and is measured explicitly as a planned extension (the efficiency study).
- **More code to maintain and more ways to be subtly wrong.** Mitigated by ADR-004 parity tests and by the invariance tests in math.md.
- Week 2–4 effort goes into code that already exists in PyTorch. That effort is the point of the project.

### Risks and mitigations
| Risk | Mitigation |
|---|---|
| A forbidden API slips into `src/` (for example a convenient `nn.LayerNorm`) | **AST-based policy test** `tests/policy/test_forbidden_apis.py`, run in CI from Week 1 |
| A hand-written component is silently wrong | Parity tests (ADR-004) plus invariance tests (math.md §1.3, §4, §8) |
| Performance is too slow for the IWSLT runs | Profile first. If needed, write an ADR for an opt-in SDPA backend, kept behind a parity test. |

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Use `nn.Transformer` and focus on training and evaluation | Removes the core learning and signalling value; ablations of internals become hard |
| Wrap Hugging Face `MarianMT`/`T5` configurations | Same problem, plus a large dependency surface; it becomes a fine-tuning project, not a reproduction |
| Implement *everything* from scratch, including `nn.Linear`, Adam and autograd | Little marginal learning for a lot of effort; it moves the focus away from the paper |
| Write it by hand but use `F.scaled_dot_product_attention` for speed | Hides exactly the component (scaling, masking, softmax) the project exists to demonstrate. Deferred to an optional backend. |

## Compliance
- `tests/policy/test_forbidden_apis.py` parses every `.py` file under `src/` with Python's `ast` module (so docstrings and comments are not false positives) and fails on any forbidden attribute access or import. It runs on every push via CI.
- Code-review checklist item: "No forbidden APIs; any reference implementation is used only in tests."
