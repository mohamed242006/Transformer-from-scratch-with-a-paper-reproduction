"""Complete models assembled from modules: the Transformer and (Week 7) the RNN baselines.

Owner: Model owner. Contract: docs/REPOSITORY.md#srcaiaynmodels.
"""

from aiayn.models.config import ModelConfig, expected_parameter_count, load_model_config
from aiayn.models.transformer import AttentionMaps, ModelOutput, Transformer, build_transformer

__all__ = [
    "AttentionMaps",
    "ModelConfig",
    "ModelOutput",
    "Transformer",
    "build_transformer",
    "expected_parameter_count",
    "load_model_config",
]
