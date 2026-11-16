"""Label-smoothed cross-entropy (paper §5.4; math.md §8; ADR-017)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["LossOutput", "label_smoothed_nll"]


@dataclass(frozen=True)
class LossOutput:
    """Sums over non-pad target tokens (normalize by the update's token count, ADR-017).

    Attributes:
        loss_sum: Smoothed loss summed over tokens; float32 scalar attached to the graph.
        nll_sum: Unsmoothed negative log-likelihood sum (for perplexity), detached.
        correct: Number of tokens whose argmax equals the target, detached.
    """

    loss_sum: torch.Tensor
    nll_sum: torch.Tensor
    correct: torch.Tensor


def label_smoothed_nll(
    logits: torch.Tensor, target: torch.Tensor, *, epsilon: float, ignore_index: int
) -> LossOutput:
    """Label-smoothed cross-entropy with the ε/K convention over all K classes.

    Shapes: logits ``[..., V]`` (any float dtype), target ``[...]`` int64.

    Per token: ``(1 − ε)·(−log p_y) + ε·mean_k(−log p_k)``. This equals
    ``F.cross_entropy(logits, target, label_smoothing=ε, ignore_index=pad, reduction="sum")``.
    ``log_softmax`` runs in at least float32, even under fp16 autocast (ADR-014).
    """
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    if logits.shape[:-1] != target.shape:
        raise ValueError(f"logits {tuple(logits.shape)} do not match target {tuple(target.shape)}")
    # At least float32 (ADR-014): promote fp16/bf16, never downcast float64.
    compute_dtype = torch.promote_types(logits.dtype, torch.float32)
    lprobs = F.log_softmax(logits.to(compute_dtype), dim=-1)  # [..., V]
    keep = target != ignore_index
    safe_target = target.masked_fill(~keep, 0).unsqueeze(-1)
    nll = -lprobs.gather(-1, safe_target).squeeze(-1)  # [...]
    if epsilon > 0.0:
        smooth = -lprobs.mean(dim=-1)  # (1/K) Σ_k −log p_k
        per_token = (1.0 - epsilon) * nll + epsilon * smooth
    else:
        per_token = nll
    loss_sum = per_token.masked_fill(~keep, 0.0).sum()
    nll_sum = nll.detach().masked_fill(~keep, 0.0).sum()
    correct = ((lprobs.detach().argmax(dim=-1) == target) & keep).sum()
    return LossOutput(loss_sum=loss_sum, nll_sum=nll_sum, correct=correct)
