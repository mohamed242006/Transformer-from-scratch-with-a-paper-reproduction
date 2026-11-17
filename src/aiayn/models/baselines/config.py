"""RNN baseline configuration (``configs/model/*.yaml`` with ``architecture: rnn``; ADR-021)
and its closed-form parameter count."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["RNNConfig", "rnn_parameter_count"]


@dataclass(frozen=True)
class RNNConfig:
    """Encoder-decoder RNN. ``vocab_size``/``pad_id`` come from the data at build time.

    The three baselines of DESIGN.md §7 are points in this space:
    ``lstm`` + ``none`` (Sutskever et al., 2014), ``gru`` + ``bahdanau`` with a
    bidirectional encoder (Bahdanau et al., 2015) and ``lstm`` + ``luong`` with a
    bidirectional encoder (Luong et al., 2015, global "general" attention).
    """

    name: str
    architecture: Literal["rnn"]
    cell: Literal["lstm", "gru"]
    attention: Literal["none", "bahdanau", "luong"]
    embedding_dim: int
    hidden_size: int
    num_layers: int
    bidirectional_encoder: bool = False
    dropout: float = 0.3
    tie_weights: bool = True
    #: Decode-length cap only: RNNs have no position table. Kept so the same
    #: compatibility check (longest sequence + 50) applies to every model.
    max_positions: int = 1024

    def __post_init__(self) -> None:
        if not re.match(r"^[a-z0-9_]+$", self.name):
            raise ValueError(f"name must be lowercase snake_case: {self.name!r}")
        for field_name in ("embedding_dim", "hidden_size", "num_layers", "max_positions"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def d_model(self) -> int:
        """Width used by the Noam schedule (``d^-0.5``): the recurrent hidden size."""
        return self.hidden_size

    @property
    def gates(self) -> int:
        return 4 if self.cell == "lstm" else 3

    @property
    def encoder_output_dim(self) -> int:
        return self.hidden_size * (2 if self.bidirectional_encoder else 1)

    @property
    def readout_inputs(self) -> int:
        """Width of the features fed to the readout layer."""
        h, he, e = self.hidden_size, self.encoder_output_dim, self.embedding_dim
        return {"none": h, "luong": h + he, "bahdanau": h + he + e}[self.attention]


def _rnn_params(gates: int, input_size: int, hidden: int, layers: int, directions: int) -> int:
    """PyTorch ``nn.LSTM``/``nn.GRU``: per layer and direction ``g·H·(in + H) + 2·g·H``."""
    total = 0
    for layer in range(layers):
        width = input_size if layer == 0 else hidden * directions
        total += directions * (gates * hidden * (width + hidden) + 2 * gates * hidden)
    return total


def rnn_parameter_count(config: RNNConfig, vocab_size: int) -> int:
    """Closed-form parameter count, checked against the built model in tests.

    ``P = V·E·(1 if tied else 3) + encoder RNN + decoder RNN + bridge + attention + readout``
    """
    v, e, h, n = vocab_size, config.embedding_dim, config.hidden_size, config.num_layers
    he, g = config.encoder_output_dim, config.gates
    total = v * e * (1 if config.tie_weights else 3)
    total += _rnn_params(g, e, h, n, 2 if config.bidirectional_encoder else 1)
    decoder_input = e + (he if config.attention == "bahdanau" else 0)
    total += _rnn_params(g, decoder_input, h, n, 1)
    if config.bidirectional_encoder:  # one bridge per state kind (h, and c for LSTM)
        total += (2 if config.cell == "lstm" else 1) * (2 * h * h + h)
    if config.attention == "luong":
        total += he * h  # W_a (no bias)
    elif config.attention == "bahdanau":
        total += h * h + (he * h + h) + h  # W_q (no bias), U_k (bias), v
    total += config.readout_inputs * e + e  # tanh readout to the embedding width
    return total
