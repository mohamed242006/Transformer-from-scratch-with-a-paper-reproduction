"""Label-smoothed loss (math.md §8, ADR-017)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aiayn.training.losses import label_smoothed_nll

PAD = 0


def inputs(vocab: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 5, vocab, generator=g, dtype=torch.float64)
    target = torch.randint(1, vocab, (3, 5), generator=g)
    target[0, 3:] = PAD
    target[2, 4] = PAD
    return logits, target


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.3])
def test_matches_pytorch_cross_entropy(epsilon: float) -> None:
    logits, target = inputs()
    out = label_smoothed_nll(logits, target, epsilon=epsilon, ignore_index=PAD)
    ref = F.cross_entropy(
        logits.reshape(-1, 11).float(),
        target.reshape(-1),
        label_smoothing=epsilon,
        ignore_index=PAD,
        reduction="sum",
    )
    torch.testing.assert_close(out.loss_sum.float(), ref, atol=1e-5, rtol=1e-6)
    nll = F.cross_entropy(
        logits.reshape(-1, 11).float(), target.reshape(-1), ignore_index=PAD, reduction="sum"
    )
    torch.testing.assert_close(out.nll_sum.float(), nll, atol=1e-5, rtol=1e-6)


def test_padding_contributes_nothing() -> None:
    logits, target = inputs()
    out = label_smoothed_nll(logits, target, epsilon=0.1, ignore_index=PAD)
    perturbed = logits.clone()
    perturbed[target == PAD] += 100.0
    again = label_smoothed_nll(perturbed, target, epsilon=0.1, ignore_index=PAD)
    torch.testing.assert_close(out.loss_sum, again.loss_sum)


def test_precision_is_at_least_float32_and_never_downcast() -> None:
    logits, target = inputs()
    half = label_smoothed_nll(logits.half(), target, epsilon=0.1, ignore_index=PAD)
    assert half.loss_sum.dtype == torch.float32 and torch.isfinite(half.loss_sum)
    # Regression: logits.float() used to downcast float64 inputs to float32.
    assert label_smoothed_nll(logits, target, epsilon=0.1, ignore_index=PAD).loss_sum.dtype == (
        torch.float64
    )


def test_gradient_is_p_minus_smoothed_target() -> None:
    """math.md §8.3: dL/dz = p − q'."""
    vocab, eps, y = 7, 0.2, 3
    z = torch.randn(1, vocab, dtype=torch.float64, requires_grad=True)
    label_smoothed_nll(z, torch.tensor([y]), epsilon=eps, ignore_index=PAD).loss_sum.backward()
    q = torch.full((vocab,), eps / vocab, dtype=torch.float64)
    q[y] += 1 - eps
    assert z.grad is not None
    torch.testing.assert_close(z.grad[0], z.softmax(-1)[0].detach() - q)


def test_loss_floor_is_the_smoothed_target_entropy() -> None:
    """math.md §8.4: the minimum of the smoothed loss is H(q'), not 0 (overfit tests!)."""
    vocab, eps, y = 10_000, 0.1, 5
    q = torch.full((vocab,), eps / vocab, dtype=torch.float64)
    q[y] += 1 - eps
    optimal = q.log().unsqueeze(0)  # logits whose softmax equals q'
    out = label_smoothed_nll(optimal, torch.tensor([y]), epsilon=eps, ignore_index=PAD)
    floor = -(q * q.log()).sum().item()
    assert out.loss_sum.item() == pytest.approx(floor, rel=1e-12)
    assert floor == pytest.approx(1.246, abs=1e-3)  # math.md §8.4 table


def test_correct_counts_only_real_tokens() -> None:
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = logits[0, 1, 4] = logits[0, 2, 1] = 5.0
    target = torch.tensor([[2, 3, PAD]])
    out = label_smoothed_nll(logits, target, epsilon=0.0, ignore_index=PAD)
    assert int(out.correct) == 1
    assert math.isfinite(float(out.nll_sum))


def test_validation() -> None:
    logits, target = inputs()
    with pytest.raises(ValueError):
        label_smoothed_nll(logits, target, epsilon=1.0, ignore_index=PAD)
    with pytest.raises(ValueError):
        label_smoothed_nll(logits, target[:, :4], epsilon=0.1, ignore_index=PAD)
