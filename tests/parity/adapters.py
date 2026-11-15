"""Bridges between our conventions and PyTorch reference implementations (ADR-004).

| Ours                                  | PyTorch                                          |
|---------------------------------------|--------------------------------------------------|
| bool mask, True = may attend (ADR-008)| MHA key_padding_mask / attn_mask: True = blocked |
| separate q/k/v ``nn.Linear``          | packed ``in_proj_weight [3d, d]``, ``in_proj_bias``|
"""

from __future__ import annotations

import torch
from torch import nn

from aiayn.modules.multihead import MultiHeadAttention

#: ADR-004 tolerance tiers for a single module.
TOLERANCE: dict[torch.dtype, dict[str, float]] = {
    torch.float64: {"atol": 1e-10, "rtol": 1e-8},
    torch.float32: {"atol": 1e-6, "rtol": 1e-5},
}


def to_torch_key_padding_mask(key_mask: torch.Tensor) -> torch.Tensor:
    """Ours ``[B, 1, 1, S]`` (True = attend) -> MHA ``[B, S]`` (True = ignore)."""
    assert key_mask.dim() == 4 and key_mask.shape[1] == key_mask.shape[2] == 1
    return ~key_mask[:, 0, 0, :]


def to_torch_attn_mask(query_key_mask: torch.Tensor) -> torch.Tensor:
    """Ours ``[1, 1, Tq, Tk]`` (True = attend) -> MHA ``[Tq, Tk]`` (True = blocked)."""
    assert query_key_mask.dim() == 4 and query_key_mask.shape[:2] == (1, 1)
    return ~query_key_mask[0, 0]


def reference_mha(ours: MultiHeadAttention) -> nn.MultiheadAttention:
    """An ``nn.MultiheadAttention`` holding exactly our weights (same dtype)."""
    dtype = ours.q_proj.weight.dtype
    ref = nn.MultiheadAttention(
        ours.d_model, ours.num_heads, dropout=0.0, bias=True, batch_first=True, dtype=dtype
    )
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight])
        )
        assert ref.in_proj_bias is not None
        ref.in_proj_bias.copy_(
            torch.cat([_bias(ours.q_proj), _bias(ours.k_proj), _bias(ours.v_proj)])
        )
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        ref.out_proj.bias.copy_(_bias(ours.out_proj))
    return ref


def _bias(layer: nn.Linear) -> torch.Tensor:
    assert layer.bias is not None
    return layer.bias


def unpack_reference_grads(ref: nn.MultiheadAttention) -> dict[str, torch.Tensor]:
    """Map the reference's packed gradients onto our parameter names."""
    assert ref.in_proj_weight.grad is not None and ref.in_proj_bias is not None
    assert ref.in_proj_bias.grad is not None
    assert ref.out_proj.weight.grad is not None and ref.out_proj.bias.grad is not None
    wq, wk, wv = ref.in_proj_weight.grad.chunk(3)
    bq, bk, bv = ref.in_proj_bias.grad.chunk(3)
    return {
        "q_proj.weight": wq,
        "k_proj.weight": wk,
        "v_proj.weight": wv,
        "q_proj.bias": bq,
        "k_proj.bias": bk,
        "v_proj.bias": bv,
        "out_proj.weight": ref.out_proj.weight.grad,
        "out_proj.bias": ref.out_proj.bias.grad,
    }


#: ADR-004 tolerance tier for stacks (errors accumulate across layers).
STACK_TOLERANCE: dict[torch.dtype, dict[str, float]] = {
    torch.float64: {"atol": 1e-10, "rtol": 1e-8},
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
}


def copy_mha_into(ours: MultiHeadAttention, ref: nn.MultiheadAttention) -> None:
    """Copy our separate q/k/v/out projections into a packed ``nn.MultiheadAttention``."""
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight])
        )
        assert ref.in_proj_bias is not None
        ref.in_proj_bias.copy_(
            torch.cat([_bias(ours.q_proj), _bias(ours.k_proj), _bias(ours.v_proj)])
        )
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        ref.out_proj.bias.copy_(_bias(ours.out_proj))


def _copy_linear(src: nn.Linear, dst: nn.Linear) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        dst.bias.copy_(_bias(src))


def _copy_norm(src: nn.Module, dst: nn.LayerNorm) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)  # type: ignore[arg-type]
        dst.bias.copy_(src.bias)  # type: ignore[arg-type]


def copy_encoder_layer(ours: nn.Module, ref: nn.TransformerEncoderLayer) -> None:
    """Map our EncoderLayer onto ``nn.TransformerEncoderLayer`` (norm1 = attn, norm2 = FFN)."""
    copy_mha_into(ours.self_attn, ref.self_attn)  # type: ignore[arg-type]
    _copy_linear(ours.ffn.linear1, ref.linear1)  # type: ignore[union-attr, arg-type]
    _copy_linear(ours.ffn.linear2, ref.linear2)  # type: ignore[union-attr, arg-type]
    _copy_norm(ours.self_attn_block.norm, ref.norm1)  # type: ignore[union-attr, arg-type]
    _copy_norm(ours.ffn_block.norm, ref.norm2)  # type: ignore[union-attr, arg-type]


def copy_decoder_layer(ours: nn.Module, ref: nn.TransformerDecoderLayer) -> None:
    """Map our DecoderLayer onto ``nn.TransformerDecoderLayer`` (norm1/2/3 = self/cross/FFN)."""
    copy_mha_into(ours.self_attn, ref.self_attn)  # type: ignore[arg-type]
    copy_mha_into(ours.cross_attn, ref.multihead_attn)  # type: ignore[arg-type]
    _copy_linear(ours.ffn.linear1, ref.linear1)  # type: ignore[union-attr, arg-type]
    _copy_linear(ours.ffn.linear2, ref.linear2)  # type: ignore[union-attr, arg-type]
    _copy_norm(ours.self_attn_block.norm, ref.norm1)  # type: ignore[union-attr, arg-type]
    _copy_norm(ours.cross_attn_block.norm, ref.norm2)  # type: ignore[union-attr, arg-type]
    _copy_norm(ours.ffn_block.norm, ref.norm3)  # type: ignore[union-attr, arg-type]


def reference_layer_kwargs(
    d_model: int, num_heads: int, d_ff: int, norm_first: bool, dtype: torch.dtype
) -> dict[str, object]:
    return {
        "d_model": d_model,
        "nhead": num_heads,
        "dim_feedforward": d_ff,
        "dropout": 0.0,
        "activation": "relu",
        "layer_norm_eps": 1e-5,
        "batch_first": True,
        "norm_first": norm_first,
        "dtype": dtype,
    }
