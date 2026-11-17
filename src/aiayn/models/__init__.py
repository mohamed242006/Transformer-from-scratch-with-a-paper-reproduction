"""Complete models assembled from modules: the Transformer and the RNN baselines.

Every model implements ``EncoderDecoder``; ``build_model`` dispatches on the config type.

Owner: Model owner. Contract: docs/REPOSITORY.md#srcaiaynmodels.
"""

from aiayn.models.base import EncoderDecoder
from aiayn.models.baselines import RNNConfig, RNNMemory, RNNSeq2Seq, build_rnn_seq2seq
from aiayn.models.config import ModelConfig, expected_parameter_count, load_model_config
from aiayn.models.factory import (
    AnyModelConfig,
    analytic_parameter_count,
    build_model,
    load_any_model_config,
    model_config_from_dict,
)
from aiayn.models.transformer import AttentionMaps, ModelOutput, Transformer, build_transformer

__all__ = [
    "AnyModelConfig",
    "AttentionMaps",
    "EncoderDecoder",
    "ModelConfig",
    "ModelOutput",
    "RNNConfig",
    "RNNMemory",
    "RNNSeq2Seq",
    "Transformer",
    "analytic_parameter_count",
    "build_model",
    "build_rnn_seq2seq",
    "build_transformer",
    "expected_parameter_count",
    "load_any_model_config",
    "load_model_config",
    "model_config_from_dict",
]
