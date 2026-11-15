from __future__ import annotations

import pytest
import torch

from aiayn.modules import Decoder, DecoderLayer
from aiayn.modules.masking import decoder_attention_mask, encoder_attention_mask

D, H, FF = 32, 4, 64
SRC = torch.tensor([[5, 6, 7, 3, 0, 0], [5, 6, 7, 8, 9, 3]])
TGT = torch.tensor([[2, 5, 6, 0], [2, 7, 8, 9]])


def tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(2, 4, D, generator=g),
        torch.randn(2, 6, D, generator=g),
        decoder_attention_mask(TGT, 0),
        encoder_attention_mask(SRC, 0),
    )


def test_layer_shapes_and_parameter_count() -> None:
    layer = DecoderLayer(D, H, FF)
    y, memory, self_mask, mem_mask = tensors()
    out, self_w, cross_w = layer(y, memory, self_mask, mem_mask)
    assert out.shape == (2, 4, D)
    assert self_w.shape == (2, H, 4, 4) and cross_w.shape == (2, H, 4, 6)
    assert sum(p.numel() for p in layer.parameters()) == 8 * D * D + 2 * D * FF + FF + 15 * D


def test_cross_attention_queries_come_from_decoder_and_respect_source_padding() -> None:
    torch.manual_seed(0)
    layer = DecoderLayer(D, H, FF).eval()
    y, memory, self_mask, mem_mask = tensors()
    _, _, cross_w = layer(y, memory, self_mask, mem_mask)
    assert torch.count_nonzero(cross_w[0, ..., 4:]) == 0  # padded source keys
    perturbed = memory.clone()
    perturbed[0, 4:] = 100.0
    base = layer(y, memory, self_mask, mem_mask)[0]
    torch.testing.assert_close(layer(y, perturbed, self_mask, mem_mask)[0], base)


def test_decoder_layer_is_causal() -> None:
    torch.manual_seed(0)
    layer = DecoderLayer(D, H, FF).eval()
    y, memory, _, mem_mask = tensors()
    tgt = torch.tensor([[2, 5, 6, 7], [2, 7, 8, 9]])
    self_mask = decoder_attention_mask(tgt, 0)
    base = layer(y, memory, self_mask, mem_mask)[0]
    for t in range(1, 4):
        perturbed = y.clone()
        perturbed[:, t:] += 25.0
        out = layer(perturbed, memory, self_mask, mem_mask)[0]
        torch.testing.assert_close(out[:, :t], base[:, :t])


@pytest.mark.parametrize("position", ["post", "pre"])
def test_stack_shapes_maps_and_final_norm(position: str) -> None:
    decoder = Decoder(3, D, H, FF, norm_position=position)  # type: ignore[arg-type]
    y, memory, self_mask, mem_mask = tensors()
    out, self_maps, cross_maps = decoder(y, memory, self_mask, mem_mask)
    assert out.shape == (2, 4, D) and self_maps == () and cross_maps == ()
    _, self_maps, cross_maps = decoder(y, memory, self_mask, mem_mask, need_weights=True)
    assert [m.shape for m in self_maps] == [(2, H, 4, 4)] * 3
    assert [m.shape for m in cross_maps] == [(2, H, 4, 6)] * 3
    assert (decoder.final_norm is None) == (position == "post")


def test_stack_validation() -> None:
    decoder = Decoder(1, D, H, FF)
    y, memory, self_mask, mem_mask = tensors()
    with pytest.raises(ValueError, match="batch"):
        decoder(y, memory[:1], self_mask, mem_mask)
    with pytest.raises(ValueError):
        decoder(y, memory[..., :8], self_mask, mem_mask)
