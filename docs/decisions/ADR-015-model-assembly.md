# ADR-015: Model assembly: boundaries, weight tying, attention-weight retention, initialization

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [ADR-003](ADR-003-post-ln-default.md), [ADR-004](ADR-004-parity-testing.md), [ADR-008](ADR-008-mask-convention.md), [ADR-011](ADR-011-initialization.md), [ADR-012](ADR-012-dropout-locations.md), [ADR-013](ADR-013-run-artifacts-and-resumption.md), [math.md §5, §9](../math.md)

## Context
Week 4 assembles the Week 3 modules into the full encoder-decoder. A review of the modules and ADR-011…014 found five decisions that cannot be left to implementation detail:
1. `SublayerConnection` passes only the sub-layer's *output*, so attention weights (needed for spec §11.3 heatmaps and per-head entropy logging) are lost unless captured on purpose.
2. A naive `model.reset_parameters()` that visits every submodule also calls `nn.Linear.reset_parameters()` (Kaiming default), silently undoing ADR-011's Xavier initialization.
3. The common "deepcopy one layer N times" pattern (Annotated Transformer) gives all N layers **identical** initial weights.
4. Shared-embedding references held by several submodules produce duplicate `state_dict` keys; a corrupted or edited checkpoint could then silently untie weights on load.
5. Exact parity against `nn.TransformerEncoder`/`nn.TransformerDecoder` (ADR-004) requires our stacks to have the same boundaries as theirs.

## Decision
1. **Boundaries.** `Encoder` and `Decoder` (in `aiayn.modules`) are pure layer stacks plus the Pre-LN final norm, mirroring `nn.TransformerEncoder(norm=None|LN)` and `nn.TransformerDecoder(...)`: they take embedded inputs and masks. `Transformer` (in `aiayn.models`) owns the token embeddings, the positional encodings (one per side), **mask construction** from `pad_id` (callers pass token IDs only), and the output projection.
2. **Independent layers.** Each layer is constructed separately (no `deepcopy`), so every layer gets its own random initialization. A test asserts this.
3. **Weight tying.** `tie_weights=True` (default, paper §3.4): one matrix, registered once as `embeddings.shared.weight`. It serves as the source embedding, the target embedding and the output projection (`logits = h Eᵀ`, **no output bias**, no √d on the output side). `tie_weights=False`: three separate matrices (`embeddings.source`, `embeddings.target`, `output_projection`, all 𝒩(0, 1/d)). One joint vocabulary is required either way (ADR-005). The `state_dict` never contains two keys for the same storage (tested).
4. **Stack-level norms.** Post-LN: none. Pre-LN: one final LayerNorm per stack (ADR-003).
5. **Attention-weight retention.** Layers always return their (pre-dropout) attention weights alongside the output. They are already computed, and autograd keeps them alive anyway, so returning them costs nothing. Stacks *collect* them only when asked (`need_weights=True`), and **detach** them, so logging never keeps computation graphs alive. Public API:
   - `Transformer.forward(src, tgt_in) -> logits [B, T, V]` (training path; nothing retained)
   - `Transformer.forward_with_attention(src, tgt_in) -> ModelOutput(logits, AttentionMaps)` where `AttentionMaps` holds per-layer tuples: `encoder_self [B,h,S,S]`, `decoder_self [B,h,T,T]`, `decoder_cross [B,h,T,S]` (float32 or wider, ADR-014).
6. **Initialization.** Constructors initialize (ADR-011). `Transformer.reset_parameters()` calls `reset_parameters()` **only on project modules** (embedding, positional, attention, FFN, LayerNorm) in registration order, never on raw `nn.Linear`. `build_transformer(config, vocab_size=…, pad_id=…, seed=…)` constructs under `torch.random.fork_rng`, so initialization is a pure function of `(config, vocab_size, seed)` and independent of global RNG state. This addresses the construction-order risk found in Week 3.
7. **Numerical safeguards.** Shape and dtype checks are always on (no GPU sync). `model.debug_checks = True` additionally validates mask invariants (ADR-008), token-ID ranges and finite logits. These checks synchronize with the GPU, so they are for debugging and the Kaggle preflight only.
8. **Configuration.** `ModelConfig` (frozen dataclass, ADR-007) holds the architecture. `vocab_size` and `pad_id` come from the data pipeline at build time, so the model package never imports `aiayn.data` (layering). `expected_parameter_count(config, vocab_size)` implements math.md §9.3 (extended for untied, Pre-LN and learned-PE variants) and must equal the counted parameters exactly.
9. **Save/load support** (not checkpointing; that is ADR-013/Week 5): the `state_dict` has unique keys and storages; the sinusoidal table is non-persistent; `model.config`, `model.vocab_size` and `model.pad_id` are enough to rebuild an identical model; loading is strict.

## Intentional deviations from DESIGN.md (all recorded)
| Item | DESIGN.md / paper | Implementation | Reason |
|---|---|---|---|
| Output projection bias | Not specified | **None** | Tied matrix only; math.md §9 counts assume it; fairseq does the same |
| Attention / activation dropout | DESIGN §2.1 lists dropout generically | 0.0 by default | ADR-012 (paper text first) |
| Positional encoding ownership | §4 lists PE as one module | One instance per side | Learned-PE ablation needs separate tables; sinusoidal costs nothing extra |
| `max_positions` | Not specified | 1024 default | Must cover max_len + 50 at inference (paper §6.1); validated at forward time |

## Consequences
**Positive:** exact layer- and stack-level parity becomes possible; tying is enforced structurally; visualizations come at zero training cost; initialization is reproducible. **Negative:** two forward entry points; layers return tuples (slightly unusual for `nn.Module`). **Risk:** someone bypasses `forward_with_attention` and expects weights → documented, and tested via `AttentionMaps` shapes.

## Alternatives considered
| Alternative | Why rejected |
|---|---|
| Forward hooks to capture weights | Implicit, easy to leave registered (memory leak), awkward with the closure-based sublayer |
| Layers store `self.last_attention` | Hidden mutable state; breaks with activation checkpointing or reentrancy; keeps tensors alive |
| `deepcopy` of a prototype layer | Identical initial weights across layers |
| Separate embedding module references inside the encoder and decoder | Duplicate `state_dict` keys; tying can silently break on load |

## Compliance
`tests/unit/test_transformer.py`, `test_encoder.py`, `test_decoder.py`, `test_model_config.py`; `tests/parity/test_parity_layers.py`, `test_parity_transformer.py`; `tests/integration/test_model_pipeline.py`.
