"""Strict, typed configuration loading: YAML -> frozen dataclasses (ADR-007).

The converter rejects unknown keys, missing required keys and type mismatches, so a
typo in a config file fails loudly instead of silently falling back to a default.

Supported annotations: ``bool``, ``int``, ``float``, ``str``, ``Any``, ``Literal[...]``,
``X | None``, nested dataclasses, ``tuple[X, ...]``, ``list[X]`` and ``dict[str, X]``.
"""

from __future__ import annotations

import copy
import dataclasses
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")

__all__ = [
    "ConfigError",
    "apply_overrides",
    "config_to_dict",
    "from_dict",
    "load_config",
    "load_yaml",
]


class ConfigError(ValueError):
    """A configuration file, mapping or override is invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file whose top level must be a mapping (an empty file yields ``{}``)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a deep copy of ``raw`` with dotted ``key=value`` overrides applied.

    Values are parsed as YAML, so ``a.b=2`` is an int, ``a.b=null`` is ``None`` and
    ``a.b='2'`` is a string. Every parent of the target key must already be a mapping.
    """
    result: dict[str, Any] = copy.deepcopy(dict(raw))
    for item in overrides:
        key, sep, value_text = item.partition("=")
        key = key.strip()
        if not sep or not key or any(not part for part in key.split(".")):
            raise ConfigError(f"override must look like 'a.b=value', got {item!r}")
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"override {item!r}: invalid YAML value: {exc}") from exc
        *parents, leaf = key.split(".")
        node: Any = result
        walked: list[str] = []
        for part in parents:
            walked.append(part)
            if not isinstance(node, dict) or not isinstance(node.get(part), dict):
                raise ConfigError(
                    f"override {item!r}: '{'.'.join(walked)}' is not an existing mapping"
                )
            node = node[part]
        node[leaf] = value
    return result


def from_dict(cls: type[T], data: Any, path: str = "") -> T:
    """Build dataclass ``cls`` from a plain mapping, validating keys and types."""
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a dataclass type")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{_where(path)}: expected a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{_where(path)}: unknown key(s) {unknown}; allowed: {sorted(fields)}")

    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        sub = _join(path, name)
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], sub)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub}: required key is missing")
    try:
        return cast(T, cls(**kwargs))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{_where(path)}: {exc}") from exc


def load_config(path: str | Path, cls: type[T], overrides: Sequence[str] = ()) -> T:
    """Load ``path``, apply ``overrides`` and convert to ``cls``; errors name the file."""
    raw = load_yaml(path)
    try:
        return from_dict(cls, apply_overrides(raw, overrides))
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize a dataclass config to YAML-friendly builtins (tuples become lists)."""
    if not (dataclasses.is_dataclass(config) and not isinstance(config, type)):
        raise TypeError(f"expected a dataclass instance, got {type(config).__name__}")
    result = _to_builtin(dataclasses.asdict(config))
    assert isinstance(result, dict)
    return result


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _where(path: str) -> str:
    return path or "<root>"


def _join(path: str, name: str | int) -> str:
    if isinstance(name, int):
        return f"{path}[{name}]"
    return f"{path}.{name}" if path else name


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)

    if tp is Any:
        return value

    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: value must not be null")
        candidates = [a for a in args if a is not type(None)]
        errors: list[str] = []
        for candidate in candidates:
            try:
                return _coerce(candidate, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: {value!r} matches none of {candidates}: {errors}")

    if value is None:
        raise ConfigError(f"{path}: value must not be null")

    if origin is Literal:
        allowed = get_args(tp)
        # Compare type as well as value so that True does not match Literal[1].
        if not any(value == a and type(value) is type(a) for a in allowed):
            raise ConfigError(f"{path}: {value!r} is not one of {list(allowed)}")
        return value

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return from_dict(tp, value, path)

    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        args = get_args(tp)
        if origin is tuple:
            if len(args) != 2 or args[1] is not Ellipsis:
                raise TypeError(f"{path}: only homogeneous tuple[X, ...] is supported")
            item_type = args[0]
        else:
            item_type = args[0] if args else Any
        items = [_coerce(item_type, v, _join(path, i)) for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items

    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        key_type, value_type = get_args(tp) or (str, Any)
        if key_type is not str:
            raise TypeError(f"{path}: only dict[str, X] is supported")
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConfigError(f"{path}: mapping keys must be strings, got {k!r}")
            out[k] = _coerce(value_type, v, _join(path, k))
        return out

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    raise TypeError(f"{path}: unsupported annotation {tp!r} in config schema")
