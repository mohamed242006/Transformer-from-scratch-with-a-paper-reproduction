"""Full-model gradient check and low-precision stability (ADR-014)."""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck
from torch.func import functional_call

from aiayn.models import build_transformer
from tests.conftest import small_model_config

pytestmark = pytest.mark.numerical

SRC = torch.tensor([[5, 6, 7, 3], [8, 3, 0, 0]])
TGT = torch.tensor([[2, 9, 10], [2, 11, 0]])


@pytest.mark.parametrize("position", ["post", "pre"])
def test_gradcheck_through_the_tied_matrix(position: str) -> None:
    """The shared matrix feeds both embeddings and the output projection; its analytic
    gradient must match finite differences through the whole model."""
    cfg = small_model_config(
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        norm_position=position,
    )
    model = build_transformer(cfg, vocab_size=12, pad_id=0, seed=0).double()
    weight = model.output_weight.detach()
    # The PAD row is excluded on purpose: padding_idx blocks its *input-side* gradient by
    # design (padded positions never reach the loss), while finite differences would still
    # see it change the (ignored) outputs at padded positions.
    pad_row = weight[:1]
    rest = weight[1:].clone().requires_grad_()

    def logits(w_rest: torch.Tensor) -> torch.Tensor:
        full = torch.cat([pad_row, w_rest])
        return functional_call(model, {"embeddings.shared.weight": full}, (SRC, TGT))

    assert gradcheck(logits, (rest,))


def test_bf16_autocast_forward_is_finite_and_close() -> None:
    model = build_transformer(small_model_config(), vocab_size=60, pad_id=0, seed=0).eval()
    reference = model(SRC, TGT)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(SRC, TGT)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.float(), reference, atol=0.1, rtol=0.1)


def test_half_precision_model_forward_is_finite() -> None:
    model = build_transformer(small_model_config(), vocab_size=60, pad_id=0, seed=0)
    model = model.half().eval()
    logits = model(SRC, TGT)
    assert logits.dtype == torch.float16 and torch.isfinite(logits).all()


def test_large_activations_do_not_overflow_post_ln() -> None:
    model = build_transformer(small_model_config(), vocab_size=60, pad_id=0, seed=0).eval()
    with torch.no_grad():
        model.src_embedding.weight.mul_(1e3)  # extreme embeddings
    model.debug_checks = True  # raises FloatingPointError on non-finite logits
    model(SRC, TGT)
