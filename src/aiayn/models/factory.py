"""Architecture dispatch: one entry point for every model config (ADR-021).

A model config with ``architecture: rnn`` is an ``RNNConfig``; anything else is the
Transformer's ``ModelConfig``. Transformer configs carry no ``architecture`` key, so their
serialized form and the experiment ``config_hash`` are unchanged by the baselines.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aiayn.models.base import EncoderDecoder
from aiayn.models.baselines.config import RNNConfig, rnn_parameter_count
from aiayn.models.baselines.rnn import build_rnn_seq2seq
from aiayn.models.config import ModelConfig, expected_parameter_count
from aiayn.models.transformer import build_transformer
from aiayn.utils.config import ConfigError, apply_overrides, from_dict, load_yaml

__all__ = [
    "AnyModelConfig",
    "analytic_parameter_count",
    "build_model",
    "load_any_model_config",
    "model_config_from_dict",
]

AnyModelConfig = ModelConfig | RNNConfig


def model_config_from_dict(raw: Any, path: str = "model") -> AnyModelConfig:
    if isinstance(raw, dict) and raw.get("architecture") == "rnn":
        return from_dict(RNNConfig, raw, path)
    return from_dict(ModelConfig, raw, path)


def load_any_model_config(path: str | Path, overrides: Sequence[str] = ()) -> AnyModelConfig:
    """Load ``configs/model/*.yaml`` of either architecture."""
    try:
        return model_config_from_dict(apply_overrides(load_yaml(path), overrides), "")
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def build_model(
    config: AnyModelConfig, *, vocab_size: int, pad_id: int, seed: int | None = None
) -> EncoderDecoder:
    if isinstance(config, RNNConfig):
        return build_rnn_seq2seq(config, vocab_size=vocab_size, pad_id=pad_id, seed=seed)
    return build_transformer(config, vocab_size=vocab_size, pad_id=pad_id, seed=seed)


def analytic_parameter_count(config: AnyModelConfig, vocab_size: int) -> int:
    if isinstance(config, RNNConfig):
        return rnn_parameter_count(config, vocab_size)
    return expected_parameter_count(config, vocab_size)
