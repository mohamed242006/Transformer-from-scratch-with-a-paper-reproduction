# ADR-008: Attention masks are boolean, True = "may attend", key-side only, broadcastable to [B, h, Tq, Tk]

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [math.md §1.3](../math.md), [ADR-004](ADR-004-parity-testing.md), [ADR-009](ADR-009-sequence-framing-and-batching.md)

## Context
Masks are where the most common silent Transformer bugs happen: future tokens leaking into the decoder, padding treated as content, NaN rows. Conventions in the ecosystem conflict. In `nn.MultiheadAttention`, True means *masked out*. In `F.scaled_dot_product_attention`, a boolean True means *takes part*. Additive float masks are a third convention. We need one convention used everywhere, written down before attention exists (Week 3).

## Decision
1. **Type:** `torch.bool`. **True = query may attend to key.** No float/additive masks are passed between modules. Conversion to fill values happens only inside attention (`masked_fill(~mask, finfo(dtype).min)`, math.md §1.3).
2. **Rank 4, broadcastable to scores `[B, h, Tq, Tk]`:**

   | Mask | Built from | Shape | Used by |
   |---|---|---|---|
   | Source padding | `src [B, S]` | `[B, 1, 1, S]` | Encoder self-attention **and** decoder cross-attention |
   | Causal | length `T` | `[1, 1, T, T]` | (combined below) |
   | Decoder self-attention | `tgt_in [B, T]` | `[B, 1, T, T]` = causal ∧ target padding | Decoder self-attention |

3. **Key-side masking only.** Padded *query* rows are **not** masked: they attend to the real keys and produce outputs that the loss ignores (`tgt_out` has PAD there). Masking padded queries as well would create fully masked rows, and therefore NaN.
4. **The causal mask includes the diagonal** (`j ≤ i`): "up to and including that position" (paper §3.2.3).
5. **Invariant:** every query row has at least one True. This is guaranteed by framing (ADR-009): every source has EOS, every decoder input starts with BOS. It is checked by `assert_valid_attention_mask` in tests and debug mode.
6. Masks are built **on the device of the token tensor**, by functions in `aiayn.modules.masking` (model semantics live in `modules/`, REPOSITORY.md). The collator does not build masks: they are cheap, and building them on the device avoids host-to-device copies of `B·T²` booleans.
7. Conversions to PyTorch's conventions exist **only** in `tests/parity/` adapters (ADR-004).

## Consequences
**Positive:** one convention, readable code (`scores.masked_fill(~mask, …)`), memory-light broadcasting, NaN-free by construction. **Negative:** a parity adapter must invert masks for `nn.MultiheadAttention`. **Risk:** someone passes a `[B, S]` mask directly → `assert_valid_attention_mask` enforces rank 4 and dtype.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| True = masked (PyTorch MHA convention) | Double negatives in code (`~mask` to keep); conflicts with SDPA's boolean convention |
| Additive float masks (0 / −inf) | Dtype-dependent (fp16 overflow), easy to add twice, NaN-prone |
| Full `[B, h, T, T]` masks | `h×` memory for no benefit |
| Masking padded queries too | Creates fully masked rows → NaN |

## Compliance
`tests/unit/test_masking.py`: shapes, dtypes, the diagonal, combination, the non-empty-row invariant, and **semantic** tests (perturbing future or padded positions leaves outputs unchanged under a reference masked softmax).
