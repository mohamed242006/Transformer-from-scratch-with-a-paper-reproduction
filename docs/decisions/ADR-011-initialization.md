# ADR-011: Weight initialization

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [math.md §5](../math.md#5-embeddings-weight-tying-and-initialization), [paper_notes.md U1](../paper_notes.md), [ADR-014](ADR-014-numerical-precision.md)

## Context
The paper does not specify initialization (paper_notes U1). Initialization interacts with three paper-exact choices: the √d embedding scale, three-way weight tying, and Post-LN (which is sensitive to early gradient scale). PyTorch's defaults (Kaiming-uniform with a = √5 for `nn.Linear`, N(0, 1) for `nn.Embedding`) are wrong for this architecture. In particular, N(0, 1) embeddings × √512 give inputs of standard deviation about 22.6, which overwhelms the positional encoding and gives huge tied logits (math.md §5.2).

## Decision
| Parameter | Initialization | Rationale |
|---|---|---|
| Shared token embedding `E ∈ ℝ^{V×d}` | 𝒩(0, d^-1), with the padding row set to 0 | After ×√d the inputs have unit variance; the same matrix used unscaled as the output projection gives logits of about unit variance (math.md §5.2–5.3). Matches tensor2tensor/fairseq. |
| Learned positional embedding (ablation) | 𝒩(0, d^-1) | fairseq convention. Note: added without scaling, so it starts weaker than the token signal (variance 1/d vs 1). |
| Attention projections W_Q, W_K, W_V, W_O | Xavier-uniform, zero bias | Preserves activation variance forward and backward |
| FFN W₁, W₂ | Xavier-uniform, zero bias | Same |
| LayerNorm γ, β | 1, 0 | Identity affine at the start |
| Sinusoidal PE | none (non-trainable buffer) | |

Rules:
1. Every module owns a `reset_parameters()` that applies the table above, and constructors call it.
2. Model-level code (Week 4) must **not** re-initialize parameters after tying (that would silently untie them or overwrite the embedding scheme). A parameter-identity test enforces this in Week 4.
3. Separate Q/K/V `nn.Linear` layers each get Xavier with fan (d, d). PyTorch's `nn.MultiheadAttention` applies Xavier to the packed `[3d, d]` matrix (a slightly smaller bound). This only matters for training-from-scratch comparisons, not parity tests, which copy weights.

## Consequences
**Positive:** unit-variance signals at the start, tied logits about 𝒩(0, 1) (initial loss ≈ ln V + 0.5, testable), and the schemes are documented and testable per module. **Negative:** the paper's exact (unpublished) scheme can't be confirmed. **Risk:** Post-LN instability if the scale is wrong → initial-loss band test (math.md §5.3) in Week 4.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| PyTorch defaults | Embedding scale is catastrophically wrong with √d scaling |
| Xavier for embeddings too | Std ≈ 0.007 for V = 37k, so inputs are about 0.17 after √d: too weak against PE (0.7) |
| T-Fixup / DeepNet scaling | Designed to remove warmup or LayerNorm; not the paper's setting. Possible extension. |

## Compliance
`tests/unit/test_init.py`: empirical standard deviations within tolerance, zero biases, padding row zero, Xavier bounds respected.
