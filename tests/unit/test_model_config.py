"""ModelConfig validation and the analytic parameter count (math.md §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiayn.models import (
    ModelConfig,
    build_transformer,
    expected_parameter_count,
    load_model_config,
)
from aiayn.utils.config import ConfigError, config_to_dict, from_dict
from aiayn.utils.params import count_parameters, parameter_breakdown
from tests.conftest import REPO_ROOT, small_model_config

CONFIGS = REPO_ROOT / "configs" / "model"

# math.md §9.4 — exact values, including the DESIGN.md §6.2 table.
DOCUMENTED = [
    ("transformer_tiny", 8_000, 7_577_600),
    ("transformer_iwslt", 10_000, 36_663_296),
    ("transformer_base", 37_000, 63_082_496),
]


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")))
def test_repository_model_configs_are_valid(path: Path) -> None:
    cfg = load_model_config(path)
    assert cfg.name == path.stem


@pytest.mark.parametrize(("name", "vocab", "expected"), DOCUMENTED)
def test_documented_parameter_counts(name: str, vocab: int, expected: int) -> None:
    cfg = load_model_config(CONFIGS / f"{name}.yaml")
    assert expected_parameter_count(cfg, vocab) == expected
    model = build_transformer(cfg, vocab_size=vocab, pad_id=0, seed=0)
    assert count_parameters(model) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"tie_weights": False},
        {"norm_position": "pre"},
        {"positional_encoding": "learned"},
        {"num_encoder_layers": 3, "num_decoder_layers": 1},
        {"tie_weights": False, "norm_position": "pre", "positional_encoding": "learned"},
    ],
)
def test_formula_matches_counted_parameters_for_every_variant(overrides: dict[str, object]) -> None:
    cfg = small_model_config(**overrides)
    model = build_transformer(cfg, vocab_size=123, pad_id=0, seed=0)
    assert count_parameters(model) == expected_parameter_count(cfg, 123)


def test_breakdown_sums_to_total() -> None:
    model = build_transformer(small_model_config(), vocab_size=100, pad_id=0, seed=0)
    breakdown = parameter_breakdown(model)
    assert set(breakdown) == {"embeddings", "encoder", "decoder"}
    assert sum(breakdown.values()) == count_parameters(model)


def test_config_round_trips_for_rebuilding() -> None:
    cfg = load_model_config(CONFIGS / "transformer_iwslt.yaml")
    assert from_dict(ModelConfig, config_to_dict(cfg)) == cfg
    assert cfg.head_dim == 128


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("num_heads=3", "divisible"),
        ("d_model=0", "positive"),
        ("dropout=1.0", r"\[0, 1\)"),
        ("attention_dropout=-0.1", r"\[0, 1\)"),
        ("norm_position=middle", "not one of"),
        ("layer_norm_eps=0.0", "positive"),
        ("name=Bad-Name", "snake_case"),
        ("d_ff=2048.5", "integer"),
    ],
)
def test_invalid_configs(override: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_model_config(CONFIGS / "transformer_tiny.yaml", (override,))


def test_odd_width_needs_learned_positions() -> None:
    with pytest.raises(ValueError, match="even"):
        small_model_config(d_model=33, num_heads=3)
    small_model_config(d_model=33, num_heads=3, positional_encoding="learned")
