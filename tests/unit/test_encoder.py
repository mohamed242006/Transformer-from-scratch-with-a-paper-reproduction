from __future__ import annotations

import pytest
import torch

from aiayn.modules import Encoder, EncoderLayer
from aiayn.modules.masking import encoder_attention_mask

D, H, FF = 32, 4, 64
TOKENS = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 7, 8, 9]])


def x_in() -> torch.Tensor:
    return torch.randn(2, 5, D, generator=torch.Generator().manual_seed(0))


def test_layer_shapes_and_parameter_count() -> None:
    layer = EncoderLayer(D, H, FF)
    out, w = layer(x_in(), encoder_attention_mask(TOKENS, 0))
    assert out.shape == (2, 5, D) and w.shape == (2, H, 5, 5)
    assert sum(p.numel() for p in layer.parameters()) == 4 * D * D + 2 * D * FF + FF + 9 * D


def test_layer_order_is_attention_then_ffn() -> None:
    """With the FFN zeroed (and Post-LN), the layer equals LN2(LN1(x + attn(x)))."""
    torch.manual_seed(0)
    layer = EncoderLayer(D, H, FF).eval()
    with torch.no_grad():
        layer.ffn.linear2.weight.zero_()
    x, mask = x_in(), encoder_attention_mask(TOKENS, 0)
    attn, _ = layer.self_attn(x, x, x, mask)
    expected = layer.ffn_block.norm(layer.self_attn_block.norm(x + attn))
    torch.testing.assert_close(layer(x, mask)[0], expected)


def test_padded_query_rows_do_not_leak_into_real_positions() -> None:
    torch.manual_seed(0)
    layer = EncoderLayer(D, H, FF).eval()
    mask = encoder_attention_mask(TOKENS, 0)
    x = x_in()
    base, _ = layer(x, mask)
    perturbed = x.clone()
    perturbed[0, 3:] = 50.0
    out, _ = layer(perturbed, mask)
    torch.testing.assert_close(out[0, :3], base[0, :3])
    torch.testing.assert_close(out[1], base[1])  # other batch element untouched


@pytest.mark.parametrize("position", ["post", "pre"])
def test_stack_layers_are_independently_initialized(position: str) -> None:
    encoder = Encoder(3, D, H, FF, norm_position=position)  # type: ignore[arg-type]
    w = [layer.self_attn.q_proj.weight for layer in encoder.layers]  # type: ignore[union-attr]
    assert not torch.equal(w[0], w[1]) and not torch.equal(w[1], w[2])
    assert (encoder.final_norm is None) == (position == "post")


def test_stack_returns_detached_maps_only_on_request() -> None:
    encoder = Encoder(2, D, H, FF)
    x = x_in().requires_grad_()
    mask = encoder_attention_mask(TOKENS, 0)
    out, maps = encoder(x, mask)
    assert out.shape == (2, 5, D) and maps == ()
    _, maps = encoder(x, mask, need_weights=True)
    assert len(maps) == 2 and all(m.shape == (2, H, 5, 5) and not m.requires_grad for m in maps)


def test_stack_validates_input() -> None:
    encoder = Encoder(1, D, H, FF)
    with pytest.raises(ValueError):
        encoder(torch.randn(5, D), encoder_attention_mask(TOKENS, 0))
    with pytest.raises(ValueError):
        encoder(torch.randn(2, 5, D + 1), encoder_attention_mask(TOKENS, 0))
    with pytest.raises(ValueError):
        Encoder(0, D, H, FF)
