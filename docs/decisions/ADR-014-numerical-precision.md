# ADR-014: Numerical precision policy (fp16 on Kaggle GPUs, fp32 reductions)

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [ADR-010](ADR-010-compute-platform.md), [ADR-008](ADR-008-mask-convention.md), [math.md §1.3, §6](../math.md)

## Context
Kaggle GPUs: **T4** (Turing, sm_75: fp16 tensor cores, **no native bf16**) and **P100** (Pascal, sm_60: no tensor cores, modest fp16 benefit). Recent PyTorch CUDA builds may drop Pascal support, so P100 availability depends on Kaggle's installed build. fp16 has a maximum of 65504 and about 3 decimal digits of precision. In a Transformer, overflow and underflow concentrate in three places: attention logits and masking (math.md §1.3), the softmax, and LayerNorm statistics (squares of large activations).

## Decision
1. **fp32 is the reference precision.** Parity tests (ADR-004) and correctness tests run in fp32/fp64.
2. **Kaggle training:** T4 → `torch.autocast(dtype=torch.float16)` + `GradScaler`. P100 → fp32, or fp16 autocast only if measured to be faster. **bf16 is not used on Kaggle GPUs**; it is allowed on CPU in tests.
3. **Module-level rules (implemented in Week 3, independent of autocast):**
   - Attention: QKᵀ runs in the input dtype (tensor cores); the scores are **upcast to at least fp32** before masking and softmax; masked positions are filled with `torch.finfo(compute_dtype).min` via `masked_fill` (never `-inf`, never `-1e9`, never additive); probabilities are cast back to the value dtype for the AV matmul.
   - LayerNorm: mean and variance are computed in **at least fp32**, two-pass (centered) variance; the output is cast back to the input dtype.
   - The √d_k scale is applied to Q before the matmul (smaller tensor, same math).
   - Parameters stay fp32 (autocast casts per operation); embeddings and the PE buffer are fp32.
4. **Loss and logits reductions in fp32** (Week 5).
5. **Tests emulate low precision on CPU:** fp16/bf16 forward passes with extreme magnitudes must stay finite and match the fp32 reference within dtype-appropriate tolerances.

## Consequences
**Positive:** about 2× throughput and half the activation memory on T4, without the classic fp16 failure modes (NaN from masked rows, overflow in variance). Behaviour doesn't depend on autocast's operation lists, which change between PyTorch versions. **Negative:** the explicit upcasts cost some memory bandwidth (scores are `[B, h, T, T]` in fp32); acceptable at sentence lengths. **Risk:** P100 not supported by the installed PyTorch → session preflight (ADR-013 §9) reports the device capability; prefer T4.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| bf16 everywhere | Not supported natively on T4/P100 |
| Rely on autocast's fp32 operation list | Implicit, version-dependent; our manual LayerNorm is not `F.layer_norm` and would not be covered |
| fp32 only | Leaves about 2× T4 throughput unused; the quota is the binding constraint |
| Additive −1e9 masks | Overflow to −inf in fp16 → NaN (math.md §1.3) |

## Compliance
`tests/numerical/`: fp16/bf16 stability of attention and LayerNorm, masked weights exactly zero in every dtype, large-magnitude LayerNorm inputs, and agreement with the fp32 reference.
