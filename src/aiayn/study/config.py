"""Study definition (``configs/study/*.yaml``; ADR-022): which runs, evaluations and
selection rules make up a multi-seed experimental study."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from aiayn.utils.config import ConfigError, load_yaml
from aiayn.utils.paths import find_repo_root, resolve_path

__all__ = ["DecodingGrid", "StudyConfig", "Variant", "VariantGroup", "load_study"]

_NAME = re.compile(r"^[a-z0-9_.]+$")


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: tuple[str, ...]
    #: Restrict to these model keys (None = every model of the group), e.g. one LR grid
    #: per model family (ADR-023).
    models: tuple[str, ...] | None = None

    def applies_to(self, model: str) -> bool:
        return self.models is None or model in self.models


@dataclass(frozen=True)
class VariantGroup:
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class DecodingGrid:
    seeds: tuple[int, ...]
    split: str
    beams: tuple[int, ...]
    length_penalties: tuple[float, ...]


@dataclass(frozen=True)
class StudyConfig:
    name: str
    reference: str
    models: dict[str, Path]  # model key -> experiment config
    main_seeds: tuple[int, ...]
    decoding: DecodingGrid
    sensitivity: VariantGroup
    ablations: VariantGroup
    bootstrap_samples: int = 1000
    default_beam: int = 4
    default_alpha: float = 0.6
    #: How the tuned training configuration is chosen (ADR-023: valid BLEU; Week 8 used
    #: valid NLL, which produced a 0.0002-nat coin flip).
    training_selection: Literal["valid_bleu", "valid_nll"] = "valid_bleu"
    #: Choose best vs averaged checkpoint per model on validation (ADR-023).
    select_checkpoint: bool = True


EMPTY_GROUP = VariantGroup((), (), ())


def _seeds(raw: Any, where: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(s, int) and s >= 0 for s in raw):
        raise ConfigError(f"{where}: expected a non-empty list of non-negative integers")
    if len(set(raw)) != len(raw):
        raise ConfigError(f"{where}: duplicate seeds")
    return tuple(raw)


def _variants(raw: Any, where: str) -> tuple[Variant, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError(f"{where}: expected a mapping of variant name -> override list")
    out = []
    for name, spec in raw.items():
        if not _NAME.match(str(name)) or name == "default":
            raise ConfigError(f"{where}.{name}: invalid variant name")
        only: tuple[str, ...] | None = None
        overrides = spec
        if isinstance(spec, Mapping):  # {overrides: [...], models: [...]}
            if set(spec) - {"overrides", "models"} or "overrides" not in spec:
                raise ConfigError(
                    f"{where}.{name}: expected keys 'overrides' and optional 'models'"
                )
            overrides = spec["overrides"]
            if "models" in spec:
                if not isinstance(spec["models"], list) or not spec["models"]:
                    raise ConfigError(f"{where}.{name}.models: expected a non-empty list")
                only = tuple(str(m) for m in spec["models"])
        if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
            raise ConfigError(f"{where}.{name}: expected a list of 'group.key=value' strings")
        if any(o.startswith("train.seed=") for o in overrides):
            raise ConfigError(f"{where}.{name}: seeds come from the study, not the variant")
        out.append(Variant(str(name), tuple(overrides), only))
    return tuple(out)


def _group(raw: Any, where: str, models: Mapping[str, Path]) -> VariantGroup:
    if raw is None:
        return EMPTY_GROUP
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: expected a mapping")
    unknown = set(raw) - {"models", "seeds", "variants"}
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    chosen = raw.get("models", "all")
    keys = tuple(models) if chosen == "all" else tuple(chosen)
    missing = [k for k in keys if k not in models]
    if missing:
        raise ConfigError(f"{where}.models: unknown model keys {missing}")
    variants = _variants(raw.get("variants"), f"{where}.variants")
    for v in variants:
        stray = [m for m in v.models or () if m not in keys]
        if stray:
            raise ConfigError(f"{where}.variants.{v.name}: models {stray} are not in this group")
    return VariantGroup(keys, _seeds(raw.get("seeds"), f"{where}.seeds"), variants)


def _choice(value: Any, allowed: tuple[str, ...], path: Path) -> Any:
    if value not in allowed:
        raise ConfigError(f"{path}: training_selection must be one of {allowed}, got {value!r}")
    return value


def load_study(path: str | Path) -> StudyConfig:
    path = Path(path)
    raw = load_yaml(path)
    allowed = {
        "name", "reference", "models", "main_seeds", "decoding", "sensitivity", "ablations",
        "bootstrap_samples", "default_beam", "default_alpha", "training_selection",
        "select_checkpoint",
    }  # fmt: skip
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    try:  # relative experiment paths are repository-relative, as in configs/experiment
        base = find_repo_root(path.resolve())
    except FileNotFoundError:  # a study file outside the repository
        base = path.resolve().parent
    models_raw = raw.get("models")
    if not isinstance(models_raw, Mapping) or not models_raw:
        raise ConfigError(f"{path}: 'models' must map model keys to experiment configs")
    models = {}
    for key, value in models_raw.items():
        if not _NAME.match(str(key)):
            raise ConfigError(f"{path}: invalid model key {key!r}")
        models[str(key)] = resolve_path(str(value), base)
    reference = str(raw.get("reference"))
    if reference not in models:
        raise ConfigError(f"{path}: reference {reference!r} is not one of {sorted(models)}")
    main_seeds = _seeds(raw.get("main_seeds"), f"{path}: main_seeds")
    d = raw.get("decoding")
    if not isinstance(d, Mapping):
        raise ConfigError(f"{path}: 'decoding' must be a mapping")
    decoding = DecodingGrid(
        seeds=_seeds(d.get("seeds"), f"{path}: decoding.seeds"),
        split=str(d.get("split", "valid")),
        beams=tuple(int(b) for b in d.get("beams", [])),
        length_penalties=tuple(float(a) for a in d.get("length_penalties", [])),
    )
    if decoding.split == "test":
        raise ConfigError(f"{path}: decoding settings must be selected on valid, never test")
    if not decoding.beams or not decoding.length_penalties or min(decoding.beams) < 1:
        raise ConfigError(f"{path}: decoding.beams and decoding.length_penalties are required")
    if not set(decoding.seeds) <= set(main_seeds):
        raise ConfigError(f"{path}: decoding.seeds must be a subset of main_seeds")
    sensitivity = _group(raw.get("sensitivity"), f"{path}: sensitivity", models)
    if not set(sensitivity.seeds) <= set(main_seeds):
        raise ConfigError(f"{path}: sensitivity.seeds must be a subset of main_seeds")
    study = StudyConfig(
        name=str(raw.get("name")),
        reference=reference,
        models=models,
        main_seeds=main_seeds,
        decoding=decoding,
        sensitivity=sensitivity,
        ablations=_group(raw.get("ablations"), f"{path}: ablations", models),
        bootstrap_samples=int(raw.get("bootstrap_samples", 1000)),
        default_beam=int(raw.get("default_beam", 4)),
        default_alpha=float(raw.get("default_alpha", 0.6)),
        training_selection=_choice(
            raw.get("training_selection", "valid_bleu"), ("valid_bleu", "valid_nll"), path
        ),
        select_checkpoint=bool(raw.get("select_checkpoint", True)),
    )
    if not _NAME.match(study.name):
        raise ConfigError(f"{path}: invalid study name {study.name!r}")
    return study
