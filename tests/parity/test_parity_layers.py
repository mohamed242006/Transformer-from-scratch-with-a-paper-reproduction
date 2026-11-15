"""Encoder/decoder layers and stacks vs ``torch.nn.Transformer*`` (ADR-004, ADR-015).

References are built without stack-level norms for Post-LN (the paper has none) and with
a final LayerNorm for Pre-LN, matching ADR-003. They run in train mode with dropout 0,
which disables PyTorch's inference fast path.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aiayn.modules import Decoder, DecoderLayer, Encoder, EncoderLayer, LayerNorm
from aiayn.modules.masking import causal_mask, decoder_attention_mask, encoder_attention_mask
from tests.parity.adapters import (
    STACK_TOLERANCE,
    TOLERANCE,
    copy_decoder_layer,
    copy_encoder_layer,
    reference_layer_kwargs,
)

pytestmark = pytest.mark.parity

D, H, FF = 32, 4, 64
PAD = 0
SRC = torch.tensor([[5, 6, 7, 8, 9, 10], [5, 6, 7, PAD, PAD, PAD], [4, 4, 4, 4, 4, PAD]])
TGT = torch.tensor([[2, 5, 6, 7, PAD], [2, 8, PAD, PAD, PAD], [2, 4, 4, 4, 4]])
DTYPES = [torch.float64, torch.float32]
POSITIONS = ["post", "pre"]


def masks() -> dict[str, torch.Tensor]:
    src_mask = encoder_attention_mask(SRC, PAD)
    tgt_mask = decoder_attention_mask(TGT, PAD)
    return {
        "ours_src": src_mask,
        "ours_tgt": tgt_mask,
        "src_kpm": ~src_mask[:, 0, 0, :],
        "tgt_kpm": ~(TGT != PAD),
        "causal": ~causal_mask(TGT.shape[1])[0, 0],
    }


def inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(3, SRC.shape[1], D, generator=g, dtype=dtype),
        torch.randn(3, TGT.shape[1], D, generator=g, dtype=dtype),
    )


def randomize_norms(module: nn.Module) -> None:
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, LayerNorm) and m.weight is not None and m.bias is not None:
                m.weight.normal_(1.0, 0.2)
                m.bias.normal_(0.0, 0.2)


def grads(module: nn.Module) -> list[torch.Tensor]:
    return [p.grad for p in module.parameters() if p.grad is not None]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("position", POSITIONS)
def test_encoder_layer_parity(dtype: torch.dtype, position: str) -> None:
    torch.manual_seed(0)
    ours = EncoderLayer(D, H, FF, norm_position=position).to(dtype)  # type: ignore[arg-type]
    randomize_norms(ours)
    ref = nn.TransformerEncoderLayer(**reference_layer_kwargs(D, H, FF, position == "pre", dtype))  # type: ignore[arg-type]
    copy_encoder_layer(ours, ref)
    m = masks()
    x, _ = inputs(dtype)
    x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
    out, _ = ours(x1, m["ours_src"])
    ref_out = ref(x2, src_key_padding_mask=m["src_kpm"])
    torch.testing.assert_close(out, ref_out, **TOLERANCE[dtype])
    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    (ref_out * upstream).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, **TOLERANCE[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("position", POSITIONS)
def test_decoder_layer_parity(dtype: torch.dtype, position: str) -> None:
    torch.manual_seed(0)
    ours = DecoderLayer(D, H, FF, norm_position=position).to(dtype)  # type: ignore[arg-type]
    randomize_norms(ours)
    ref = nn.TransformerDecoderLayer(**reference_layer_kwargs(D, H, FF, position == "pre", dtype))  # type: ignore[arg-type]
    copy_decoder_layer(ours, ref)
    m = masks()
    memory, y = inputs(dtype)
    y1, y2 = y.clone().requires_grad_(), y.clone().requires_grad_()
    mem1, mem2 = memory.clone().requires_grad_(), memory.clone().requires_grad_()
    out, _, _ = ours(y1, mem1, m["ours_tgt"], m["ours_src"])
    ref_out = ref(
        y2,
        mem2,
        tgt_mask=m["causal"],
        tgt_key_padding_mask=m["tgt_kpm"],
        memory_key_padding_mask=m["src_kpm"],
    )
    torch.testing.assert_close(out, ref_out, **TOLERANCE[dtype])
    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    (ref_out * upstream).sum().backward()
    torch.testing.assert_close(y1.grad, y2.grad, **TOLERANCE[dtype])
    torch.testing.assert_close(mem1.grad, mem2.grad, **TOLERANCE[dtype])


def _final_norm(ours: Encoder | Decoder, dtype: torch.dtype) -> nn.LayerNorm | None:
    if ours.final_norm is None:
        return None
    norm = nn.LayerNorm(D, eps=1e-5, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(ours.final_norm.weight)  # type: ignore[arg-type]
        norm.bias.copy_(ours.final_norm.bias)  # type: ignore[arg-type]
    return norm


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("position", POSITIONS)
def test_encoder_stack_parity(dtype: torch.dtype, position: str) -> None:
    torch.manual_seed(0)
    ours = Encoder(3, D, H, FF, norm_position=position).to(dtype)  # type: ignore[arg-type]
    randomize_norms(ours)
    layer = nn.TransformerEncoderLayer(**reference_layer_kwargs(D, H, FF, position == "pre", dtype))  # type: ignore[arg-type]
    ref = nn.TransformerEncoder(layer, 3, norm=_final_norm(ours, dtype), enable_nested_tensor=False)
    for our_layer, ref_layer in zip(ours.layers, ref.layers, strict=True):
        copy_encoder_layer(our_layer, ref_layer)  # type: ignore[arg-type]
    m = masks()
    x, _ = inputs(dtype)
    out, _ = ours(x, m["ours_src"])
    ref_out = ref(x, src_key_padding_mask=m["src_kpm"])
    real = SRC != PAD  # PyTorch may return zeros at padded positions; compare real ones
    torch.testing.assert_close(out[real], ref_out[real], **STACK_TOLERANCE[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("position", POSITIONS)
def test_decoder_stack_parity_with_gradients(dtype: torch.dtype, position: str) -> None:
    torch.manual_seed(0)
    ours = Decoder(3, D, H, FF, norm_position=position).to(dtype)  # type: ignore[arg-type]
    randomize_norms(ours)
    layer = nn.TransformerDecoderLayer(**reference_layer_kwargs(D, H, FF, position == "pre", dtype))  # type: ignore[arg-type]
    ref = nn.TransformerDecoder(layer, 3, norm=_final_norm(ours, dtype))
    for our_layer, ref_layer in zip(ours.layers, ref.layers, strict=True):
        copy_decoder_layer(our_layer, ref_layer)  # type: ignore[arg-type]
    m = masks()
    memory, y = inputs(dtype)
    out, _, _ = ours(y, memory, m["ours_tgt"], m["ours_src"])
    ref_out = ref(
        y,
        memory,
        tgt_mask=m["causal"],
        tgt_key_padding_mask=m["tgt_kpm"],
        memory_key_padding_mask=m["src_kpm"],
    )
    torch.testing.assert_close(out, ref_out, **STACK_TOLERANCE[dtype])
    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    (ref_out * upstream).sum().backward()
    # FFN weights of every layer: gradients must match layer by layer.
    for our_layer, ref_layer in zip(ours.layers, ref.layers, strict=True):
        torch.testing.assert_close(
            our_layer.ffn.linear1.weight.grad,  # type: ignore[union-attr]
            ref_layer.linear1.weight.grad,
            **STACK_TOLERANCE[dtype],
        )
