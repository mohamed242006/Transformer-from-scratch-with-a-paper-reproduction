"""Expand a study into runs and evaluations, and apply its selection rules (ADR-022).

Phases, in priority order (a session that runs out of time leaves the most important
results complete):

1. ``main``: every model, ``main_seeds``, protocol defaults; test evaluations of the
   averaged and best checkpoints (beam 4, α 0.6) and best greedy.
2. ``decoding``: on main runs with ``decoding.seeds``, the beam × α grid on **valid**.
3. ``decoding_selected``: once a model's grid is complete, its valid-best (beam, α) is
   evaluated on test for all main seeds.
4. ``ablations`` and 5. ``sensitivity``: variants × seeds, test (averaged, beam 4).
6. ``tuned``: once a model's sensitivity grid is complete, the variant with the lowest
   mean **validation** NLL gets the remaining main seeds.

Variants that resolve to the model's default config (same config hash) are aliases of
the main runs and are never trained twice. Test data never drives a selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from aiayn.inference.decoding import DecodeConfig
from aiayn.inference.evaluate import eval_tag
from aiayn.study.config import StudyConfig, Variant
from aiayn.study.storage import RunState
from aiayn.training.experiment import load_experiment

__all__ = [
    "EvalSpec",
    "PlannedRun",
    "Task",
    "all_run_names",
    "experiment_name",
    "is_alias",
    "main_run",
    "planned_runs",
    "select_decoding",
    "select_training",
    "tasks",
]

Group = Literal["main", "ablation", "sensitivity", "tuned"]
DEFAULT = "default"


@dataclass(frozen=True)
class EvalSpec:
    split: str
    checkpoint: str
    beam: int
    alpha: float

    @property
    def decode(self) -> DecodeConfig:
        return DecodeConfig(beam_size=self.beam, length_penalty=self.alpha)

    @property
    def tag(self) -> str:
        return eval_tag(self.split, self.checkpoint, self.decode)


@dataclass(frozen=True)
class PlannedRun:
    name: str
    model: str
    group: Group
    variant: str
    seed: int
    experiment: Path
    overrides: tuple[str, ...]  # variant overrides + train.seed

    @property
    def keep_best(self) -> bool:
        return self.group == "main"


@dataclass(frozen=True)
class Task:
    run: PlannedRun
    train: bool
    evals: tuple[EvalSpec, ...]
    phase: str


@cache
def _experiment(path: Path, overrides: tuple[str, ...]) -> tuple[str, str]:
    e = load_experiment(path, overrides)
    return e.name, e.config_hash


def experiment_name(study: StudyConfig, model: str) -> str:
    return _experiment(study.models[model], ())[0]


def is_alias(study: StudyConfig, model: str, variant: Variant) -> bool:
    """True if the variant does not change the model's resolved default config."""
    path = study.models[model]
    return (
        _experiment(path, (*variant.overrides, "train.seed=1"))[1]
        == _experiment(path, ("train.seed=1",))[1]
    )


def _run(
    study: StudyConfig, model: str, group: Group, variant: Variant | None, seed: int
) -> PlannedRun:
    base = experiment_name(study, model)
    name = f"{base}_s{seed}" if variant is None else f"{base}-{variant.name}_s{seed}"
    overrides = (*(variant.overrides if variant else ()), f"train.seed={seed}")
    return PlannedRun(
        name,
        model,
        group,
        variant.name if variant else DEFAULT,
        seed,
        study.models[model],
        overrides,
    )


def main_run(study: StudyConfig, model: str, seed: int) -> PlannedRun:
    return _run(study, model, "main", None, seed)


def variant_run(
    study: StudyConfig, model: str, group: Group, variant: Variant, seed: int
) -> PlannedRun:
    return _run(study, model, group, variant, seed)


def planned_runs(study: StudyConfig) -> list[PlannedRun]:
    """Static runs: main, then ablations, then sensitivity (aliases skipped)."""
    runs = [main_run(study, m, s) for m in study.models for s in study.main_seeds]
    for group_name, group in (("ablation", study.ablations), ("sensitivity", study.sensitivity)):
        for model in group.models:
            for variant in group.variants:
                if is_alias(study, model, variant):
                    continue
                runs += [variant_run(study, model, group_name, variant, s) for s in group.seeds]  # type: ignore[arg-type]
    return runs


def all_run_names(study: StudyConfig) -> list[str]:
    """Every run name the study may use, including potential tuned seeds."""
    names = [r.name for r in planned_runs(study)]
    for model in study.sensitivity.models:
        for variant in study.sensitivity.variants:
            names += [variant_run(study, model, "tuned", variant, s).name for s in study.main_seeds]
    return sorted(set(names))


# ----------------------------------------------------------------------------- evaluations
def default_test(study: StudyConfig, checkpoint: str = "averaged") -> EvalSpec:
    return EvalSpec("test", checkpoint, study.default_beam, study.default_alpha)


def main_evals(study: StudyConfig) -> tuple[EvalSpec, ...]:
    return (
        default_test(study, "averaged"),
        default_test(study, "best"),
        EvalSpec("test", "best", 1, study.default_alpha),
    )


def grid_evals(study: StudyConfig) -> tuple[EvalSpec, ...]:
    d = study.decoding
    specs = {}
    for beam in d.beams:
        for alpha in d.length_penalties if beam > 1 else (study.default_alpha,):
            spec = EvalSpec(d.split, "averaged", beam, alpha)
            specs[spec.tag] = spec  # greedy appears once
    return tuple(specs.values())


# ----------------------------------------------------------------------------- selection
def _state(states: Mapping[str, RunState], name: str) -> RunState | None:
    return states.get(name)


def select_decoding(
    study: StudyConfig, model: str, states: Mapping[str, RunState]
) -> EvalSpec | None:
    """Valid-best (beam, α) by mean BLEU over ``decoding.seeds``; None until complete.

    Ties keep the protocol default, then the smaller beam.
    """
    grid = grid_evals(study)
    totals: dict[str, float] = {spec.tag: 0.0 for spec in grid}
    for seed in study.decoding.seeds:
        state = _state(states, main_run(study, model, seed).name)
        if state is None or state.failed:
            continue  # failed seeds do not block the selection
        for spec in grid:
            result = state.evals.get(spec.tag)
            if result is None:
                return None
            totals[spec.tag] += float(result["bleu"])
    default_tag = EvalSpec(
        study.decoding.split, "averaged", study.default_beam, study.default_alpha
    ).tag
    ranked = sorted(
        grid, key=lambda s: (-round(totals[s.tag], 9), s.tag != default_tag, s.beam, s.alpha)
    )
    best = ranked[0]
    return EvalSpec("test", "averaged", best.beam, best.alpha)


def select_training(study: StudyConfig, model: str, states: Mapping[str, RunState]) -> str | None:
    """Variant (or ``default``) with the lowest mean best validation NLL over the
    sensitivity seeds; failed runs count as +inf. None until every run has finished."""
    group = study.sensitivity
    if model not in group.models:
        return DEFAULT
    candidates: dict[str, list[str]] = {
        DEFAULT: [main_run(study, model, s).name for s in group.seeds]
    }
    for variant in group.variants:
        if not is_alias(study, model, variant):
            candidates[variant.name] = [
                variant_run(study, model, "sensitivity", variant, s).name for s in group.seeds
            ]
    means: dict[str, float] = {}
    for variant_name, names in candidates.items():
        values = []
        for name in names:
            state = _state(states, name)
            if state is None or not (state.completed or state.failed):
                return None
            nll = state.best_valid_nll
            values.append(float("inf") if state.failed or nll is None else float(nll))
        means[variant_name] = sum(values) / len(values)
    return min(means, key=lambda v: (means[v], v != DEFAULT))


# ----------------------------------------------------------------------------- tasks
def _task(
    run: PlannedRun, states: Mapping[str, RunState], wanted: tuple[EvalSpec, ...], phase: str
) -> Task | None:
    state = _state(states, run.name)
    if state is not None and state.failed:
        return None
    train = state is None or not state.completed
    have = set(state.evals) if state is not None else set()
    missing = tuple(spec for spec in wanted if spec.tag not in have)
    if not train and not missing:
        return None
    return Task(run, train, missing, phase)


def tasks(study: StudyConfig, states: Mapping[str, RunState]) -> list[Task]:
    """Everything still to do, in priority order (recomputed after every task)."""
    todo: list[Task | None] = []
    for model in study.models:
        for seed in study.main_seeds:
            todo.append(_task(main_run(study, model, seed), states, main_evals(study), "main"))
    for model in study.models:
        for seed in study.decoding.seeds:
            run = main_run(study, model, seed)
            state = _state(states, run.name)
            if state is not None and state.usable:
                todo.append(_task(run, states, grid_evals(study), "decoding"))
    for model in study.models:
        selected = select_decoding(study, model, states)
        if selected is None or selected.tag == default_test(study).tag:
            continue
        for seed in study.main_seeds:
            state = _state(states, main_run(study, model, seed).name)
            if state is not None and state.usable:
                todo.append(
                    _task(main_run(study, model, seed), states, (selected,), "decoding_selected")
                )
    static = planned_runs(study)
    for group in ("ablation", "sensitivity"):
        for run in static:
            if run.group == group:
                todo.append(_task(run, states, (default_test(study),), group))
    by_name = {v.name: v for v in study.sensitivity.variants}
    for model in study.sensitivity.models:
        choice = select_training(study, model, states)
        if choice in (None, DEFAULT):
            continue
        assert choice is not None
        for seed in study.main_seeds:
            if seed in study.sensitivity.seeds:
                continue
            run = variant_run(study, model, "tuned", by_name[choice], seed)
            todo.append(_task(run, states, (default_test(study),), "tuned"))
    return [t for t in todo if t is not None]
