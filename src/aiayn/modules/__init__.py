"""Hand-written Transformer building blocks.

Embeddings, positional encoding, attention, LayerNorm, FFN, masks, sub-layer connection,
encoder and decoder layers/stacks.

Owner: Model owner. Contract: docs/REPOSITORY.md#srcaiaynmodules.
"""

from aiayn.modules.attention import scaled_dot_product_attention
from aiayn.modules.decoder import Decoder, DecoderLayer
from aiayn.modules.embedding import TokenEmbedding
from aiayn.modules.encoder import Encoder, EncoderLayer
from aiayn.modules.feedforward import PositionwiseFeedForward
from aiayn.modules.layernorm import LayerNorm
from aiayn.modules.masking import (
    assert_valid_attention_mask,
    causal_mask,
    decoder_attention_mask,
    encoder_attention_mask,
    padding_mask,
)
from aiayn.modules.multihead import MultiHeadAttention, merge_heads, split_heads
from aiayn.modules.positional import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEncoding,
    sinusoidal_table,
)
from aiayn.modules.sublayer import NormPosition, SublayerConnection

__all__ = [
    "Decoder",
    "DecoderLayer",
    "Encoder",
    "EncoderLayer",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "MultiHeadAttention",
    "NormPosition",
    "PositionwiseFeedForward",
    "SinusoidalPositionalEncoding",
    "SublayerConnection",
    "TokenEmbedding",
    "assert_valid_attention_mask",
    "causal_mask",
    "decoder_attention_mask",
    "encoder_attention_mask",
    "merge_heads",
    "padding_mask",
    "scaled_dot_product_attention",
    "sinusoidal_table",
    "split_heads",
]
