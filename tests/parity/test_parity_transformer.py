"""Full-model parity: our Transformer vs a reference assembled from PyTorch's stacks.

Reference = closed-form embedding (lookup·√d) + closed-form sinusoids (computed here,
independently of ``sinusoidal_table``) + ``nn.TransformerEncoder``/``nn.TransformerDecoder``
with copied weights + ``h Eᵀ``. Checks logits and the gradient of the shared matrix.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from aiayn.models import ModelConfig, Transformer, build_transformer
from tests.parity.adapters import (
    STACK_TOLERANCE,
    copy_decoder_layer,
    copy_encoder_layer,
    reference_layer_kwargs,
)

pytestmark = pytest.mark.parity

PAD, V = 0, 40
SRC = torch.tensor([[5, 6, 7, 8, 9, 3], [5, 6, 3, PAD, PAD, PAD]])
TGT = torch.tensor([[2, 11, 12, 13], [2, 14, PAD, PAD]])


def closed_form_pe(length: int, d: int, dtype: torch.dtype) -> torch.Tensor:
    pe = torch.zeros(length, d, dtype=torch.float64)
    for pos in range(length):
        for i in range(d // 2):
            angle = pos / 10000 ** (2 * i / d)
            pe[pos, 2 * i], pe[pos, 2 * i + 1] = math.sin(angle), math.cos(angle)
    return pe.to(dtype)


def reference_logits(model: Transformer, weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    cfg = model.config
    pre = cfg.norm_position == "pre"
    kwargs = reference_layer_kwargs(cfg.d_model, cfg.num_heads, cfg.d_ff, pre, dtype)

    def final_norm(ours_norm: nn.Module | None) -> nn.LayerNorm | None:
        if ours_norm is None:
            return None
        norm = nn.LayerNorm(cfg.d_model, eps=1e-5, dtype=dtype)
        with torch.no_grad():
            norm.weight.copy_(ours_norm.weight)  # type: ignore[arg-type]
            norm.bias.copy_(ours_norm.bias)  # type: ignore[arg-type]
        return norm

    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(**kwargs),  # type: ignore[arg-type]
        cfg.num_encoder_layers,
        norm=final_norm(model.encoder.final_norm),
        enable_nested_tensor=False,
    )
    decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(**kwargs),  # type: ignore[arg-type]
        cfg.num_decoder_layers,
        norm=final_norm(model.decoder.final_norm),
    )
    for ours, ref in zip(model.encoder.layers, encoder.layers, strict=True):
        copy_encoder_layer(ours, ref)  # type: ignore[arg-type]
    for ours, ref in zip(model.decoder.layers, decoder.layers, strict=True):
        copy_decoder_layer(ours, ref)  # type: ignore[arg-type]

    scale = math.sqrt(cfg.d_model)
    # padding_idx: the PAD row gets no gradient from input lookups (only from the output side)
    src_x = F.embedding(SRC, weight, padding_idx=PAD) * scale + closed_form_pe(
        SRC.shape[1], cfg.d_model, dtype
    )
    tgt_x = F.embedding(TGT, weight, padding_idx=PAD) * scale + closed_form_pe(
        TGT.shape[1], cfg.d_model, dtype
    )
    memory = encoder(src_x, src_key_padding_mask=SRC == PAD)
    causal = torch.triu(torch.ones(TGT.shape[1], TGT.shape[1], dtype=torch.bool), diagonal=1)
    hidden = decoder(
        tgt_x,
        memory,
        tgt_mask=causal,
        tgt_key_padding_mask=TGT == PAD,
        memory_key_padding_mask=SRC == PAD,
    )
    return hidden @ weight.T


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("position", ["post", "pre"])
def test_full_model_logits_and_gradients(dtype: torch.dtype, position: str) -> None:
    cfg = ModelConfig(
        name="parity",
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        norm_position=position,  # type: ignore[arg-type]
    )
    model = build_transformer(cfg, vocab_size=V, pad_id=PAD, seed=0).to(dtype).train()
    logits = model(SRC, TGT)

    weight = model.output_weight.detach().clone().requires_grad_()
    ref_logits = reference_logits(model, weight, dtype)
    torch.testing.assert_close(logits, ref_logits, **STACK_TOLERANCE[dtype])

    upstream = torch.randn_like(logits)
    (logits * upstream).sum().backward()
    (ref_logits * upstream).sum().backward()
    # Shared matrix gets gradient from both embeddings and the output projection.
    torch.testing.assert_close(model.output_weight.grad, weight.grad, **STACK_TOLERANCE[dtype])
