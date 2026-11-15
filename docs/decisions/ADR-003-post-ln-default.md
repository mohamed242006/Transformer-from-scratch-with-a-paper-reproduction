# ADR-003: Post-LN is the default normalization placement

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Lead engineer
- **Related:** [math.md §6.4, §7.4](../math.md), [paper_notes.md §4.1](../paper_notes.md), [ADR-004](ADR-004-parity-testing.md)

## Context

The paper states the sub-layer composition explicitly (§3.1):

$$\text{output} = \operatorname{LayerNorm}\big(x + \operatorname{Dropout}(\operatorname{Sublayer}(x))\big) \quad \text{(Post-LN)}$$

Most modern implementations, and the widely copied *Annotated Transformer*, use **Pre-LN** instead:

$$\text{output} = x + \operatorname{Dropout}(\operatorname{Sublayer}(\operatorname{LayerNorm}(x))), \quad \text{plus a final LayerNorm after the stack}$$

The trade-off, analysed in math.md §6.4 (after Xiong et al., 2020):
- **Post-LN** puts a LayerNorm Jacobian on every step of the residual path. Gradients near the output are large at initialization, so training **depends on learning-rate warmup** and can diverge with aggressive learning rates or greater depth. When it trains well, it often reaches slightly better final quality on MT at moderate depth (a commonly reported observation, not a guarantee).
- **Pre-LN** keeps an exact identity path through the residual stream, so it is more stable, less sensitive to warmup, and the norm used for deep LLMs today.

This is a reproduction project, so faithfulness to the paper is the primary criterion. Understanding *why the field moved away from Post-LN* is also one of the most valuable things the project can demonstrate.

## Decision

1. Normalization placement is a configuration option `norm_position ∈ {"post", "pre"}` implemented in **one** sublayer-connection module, so that both variants share all other code.
2. **Default: `"post"`**, which is paper-exact. In Post-LN mode there is **no** extra stack-level LayerNorm (the paper's formula has none).
3. In `"pre"` mode, a final LayerNorm is applied after the last encoder layer and after the last decoder layer (standard Pre-LN practice).
4. All **headline results** use Post-LN. Pre-LN appears in the ablation table, together with an explicit **warmup-sensitivity experiment**: Post-LN vs Pre-LN × warmup ∈ {0 (or minimal), 4000}, with per-layer gradient norms logged.
5. Parity tests (ADR-004) cover **both** modes against `nn.TransformerEncoderLayer`/`nn.TransformerDecoderLayer` with `norm_first=False` and `norm_first=True` respectively.

## Consequences

### Positive
- Faithful to the paper. Results are attributable to the paper's architecture, not a later variant.
- Produces one of the strongest analysis results in the portfolio: an *observed* demonstration of Post-LN warmup dependence, with gradient-norm plots that connect theory (math.md §6.4) to experiment.
- One flag gives a clean A/B comparison, with no forked code paths.

### Negative / costs
- Post-LN is **more fragile**. Early experiments may diverge for reasons unrelated to bugs, which makes debugging harder. Mitigation below.
- Learning-rate and warmup settings matter more, so tuning takes longer for the baselines' fair-tuning budget as well.
- Readers who know the Annotated Transformer may expect Pre-LN. The README must point out the difference.

### Risks and mitigations
| Risk | Mitigation |
|---|---|
| Divergence mistaken for an implementation bug | Debugging protocol: (1) run the same configuration with `norm_position="pre"`; if Pre-LN trains and Post-LN doesn't, suspect optimization, not code. (2) Parity tests must pass in both modes before any training is trusted. |
| Post-LN unstable on the small IWSLT data with the chosen LR | Use the paper's warmup of 4000 as the default. Log gradient norms per layer from step 1. Keep a NaN/Inf guard. |
| Silent extra LayerNorm in Post-LN mode (copied from Pre-LN code) | Parameter-count test against math.md §9 (which counts no final norms for Post-LN) |

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Pre-LN only | Not faithful to the paper; loses the warmup/stability experiment |
| Post-LN only | Loses the comparison that explains modern practice |
| Pre-LN default, Post-LN as ablation | Inverts the project's priorities. The reproduction should be the default. |
| DeepNorm / Admin / ReZero variants | Out of scope for the core; possible extension in the "2017 → modern" ablation ladder |
| RMSNorm | Modern replacement for LayerNorm, not placement. Extension item. |

## Compliance
- Configuration schema default: `norm_position: post`.
- Parity tests for both modes (ADR-004).
- Parameter-count test distinguishes the two modes (Pre-LN has +4d parameters).
- The results table labels each row with its normalization placement.
