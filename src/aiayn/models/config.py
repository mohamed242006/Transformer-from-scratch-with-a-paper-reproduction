"""Model configuration schema (``configs/model/*.yaml``; ADR-007) and the analytic
parameter count (math.md §9.3)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aiayn.utils.config import load_config

__all__ = ["ModelConfig", "expected_parameter_count", "load_model_config"]


@dataclass(frozen=True)
class ModelConfig:
    """Transformer architecture. ``vocab_size``/``pad_id`` come from the data at build time.

    Defaults are paper-exact where the paper specifies a value (ADR-012 for dropout).
    """

    name: str
    d_model: int
    num_heads: int
    d_ff: int
    num_encoder_layers: int
    num_decoder_layers: int
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    norm_position: Literal["post", "pre"] = "post"
    positional_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    max_positions: int = 1024
    tie_weights: bool = True
    scale_embedding: bool = True
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if not re.match(r"^[a-z0-9_]+$", self.name):
            raise ValueError(f"name must be lowercase snake_case: {self.name!r}")
        for field_name in (
            "d_model",
            "num_heads",
            "d_ff",
            "num_encoder_layers",
            "num_decoder_layers",
            "max_positions",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by num_heads {self.num_heads}"
            )
        if self.positional_encoding == "sinusoidal" and self.d_model % 2:
            raise ValueError("sinusoidal positional encoding needs an even d_model")
        for field_name in ("dropout", "attention_dropout", "activation_dropout"):
            p = getattr(self, field_name)
            if not 0.0 <= p < 1.0:
                raise ValueError(f"{field_name} must be in [0, 1), got {p}")
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


def load_model_config(path: str | Path, overrides: tuple[str, ...] = ()) -> ModelConfig:
    """Load and validate a ``configs/model/*.yaml`` file."""
    return load_config(path, ModelConfig, overrides)


def expected_parameter_count(config: ModelConfig, vocab_size: int) -> int:
    """Closed-form parameter count (math.md §9), extended for every config variant.

    ``P = N_enc(4d² + 2·d·d_ff + d_ff + 9d) + N_dec(8d² + 2·d·d_ff + d_ff + 15d)
         + V·d·(1 if tied else 3) [+ 4d Pre-LN final norms] [+ 2·max_pos·d learned PE]``

    Biases everywhere except the output projection (ADR-015).
    """
    d, d_ff = config.d_model, config.d_ff
    encoder_layer = 4 * d * d + 2 * d * d_ff + d_ff + 9 * d
    decoder_layer = 8 * d * d + 2 * d * d_ff + d_ff + 15 * d
    total = config.num_encoder_layers * encoder_layer + config.num_decoder_layers * decoder_layer
    total += vocab_size * d * (1 if config.tie_weights else 3)
    if config.norm_position == "pre":
        total += 2 * (2 * d)
    if config.positional_encoding == "learned":
        total += 2 * config.max_positions * d
    return total
