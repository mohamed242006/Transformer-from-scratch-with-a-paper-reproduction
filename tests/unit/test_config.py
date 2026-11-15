from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from aiayn.data.config import DataConfig, load_data_config
from aiayn.utils.config import (
    ConfigError,
    apply_overrides,
    config_to_dict,
    from_dict,
    load_config,
    load_yaml,
)
from tests.conftest import REPO_ROOT


@dataclass(frozen=True)
class Inner:
    rate: float
    name: str = "x"


@dataclass(frozen=True)
class Outer:
    count: int
    flag: bool
    mode: Literal["a", "b"]
    inner: Inner
    items: tuple[Inner, ...] = ()
    tags: list[str] = field(default_factory=list)
    table: dict[str, int] = field(default_factory=dict)
    maybe: int | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be >= 0")


VALID = {"count": 2, "flag": True, "mode": "a", "inner": {"rate": 1}}


def test_from_dict_builds_nested_typed_objects() -> None:
    cfg = from_dict(
        Outer,
        {**VALID, "items": [{"rate": 0.5, "name": "y"}], "tags": ["t"], "table": {"k": 1}},
    )
    assert cfg.inner == Inner(rate=1.0)
    assert isinstance(cfg.inner.rate, float)  # int accepted and widened for float fields
    assert cfg.items == (Inner(0.5, "y"),)
    assert cfg.tags == ["t"] and cfg.table == {"k": 1} and cfg.maybe is None


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"cuont": 1}, "unknown key"),
        ({"count": "2"}, "expected an integer"),
        ({"count": True}, "expected an integer"),  # bool is never an int
        ({"flag": 1}, "expected a boolean"),
        ({"mode": "c"}, "is not one of"),
        ({"inner": {"rate": "fast"}}, "inner.rate"),
        ({"inner": {"rate": 1, "extra": 0}}, "inner: unknown key"),
        ({"items": [{"rate": 1}, {"name": "z"}]}, "items[1].rate: required"),
        ({"maybe": "x"}, "maybe"),
        ({"count": None}, "must not be null"),
        ({"count": -1}, "count must be >= 0"),
        ({"table": {"k": "v"}}, "table.k"),
    ],
)
def test_from_dict_rejects_invalid_input(patch: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        from_dict(Outer, {**VALID, **patch})


def test_missing_required_key_is_reported_with_path() -> None:
    with pytest.raises(ConfigError, match="flag: required key is missing"):
        from_dict(Outer, {"count": 1, "mode": "a", "inner": {"rate": 1}})


def test_round_trip_through_dict() -> None:
    cfg = from_dict(Outer, {**VALID, "items": [{"rate": 2.0}]})
    assert from_dict(Outer, config_to_dict(cfg)) == cfg
    assert isinstance(config_to_dict(cfg)["items"], list)  # YAML-friendly


def test_overrides_parse_yaml_values() -> None:
    raw = {"a": {"b": 1, "c": None}, "d": "x"}
    out = apply_overrides(raw, ["a.b=2", "a.c=null", "d='5'", "a.e=[1, 2]"])
    assert out == {"a": {"b": 2, "c": None, "e": [1, 2]}, "d": "5"}
    assert raw["a"]["b"] == 1  # input not mutated


@pytest.mark.parametrize("bad", ["noequals", "=1", "a..b=1", "missing.b=1", "d.x=1"])
def test_overrides_reject_malformed_or_unknown_parents(bad: str) -> None:
    with pytest.raises(ConfigError):
        apply_overrides({"a": {"b": 1}, "d": "x"}, [bad])


def test_load_yaml_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "nope.yaml")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(tmp_path / "list.yaml")
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    assert load_yaml(tmp_path / "empty.yaml") == {}


def test_load_config_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("count: 1\nflag: true\nmode: z\ninner: {rate: 1}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"c\.yaml: mode"):
        load_config(path, Outer)
    assert load_config(path, Outer, ["mode=b"]).mode == "b"


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs" / "data").glob("*.yaml")))
def test_repository_data_configs_are_valid(path: Path) -> None:
    cfg = load_data_config(path)
    assert isinstance(cfg, DataConfig)
    assert path.stem == cfg.name


def test_data_config_cross_field_validation() -> None:
    base = load_data_config(REPO_ROOT / "configs" / "data" / "copy_task.yaml")
    raw = config_to_dict(base)
    raw["task"] = "translation"
    with pytest.raises(ConfigError, match="requires a 'translation' section"):
        from_dict(DataConfig, raw)


def test_tokenizer_special_ids_are_enforced() -> None:
    path = REPO_ROOT / "configs" / "data" / "multi30k.yaml"
    with pytest.raises(ConfigError, match="special IDs"):
        load_data_config(path, ("translation.tokenizer.pad_id=-1",))
