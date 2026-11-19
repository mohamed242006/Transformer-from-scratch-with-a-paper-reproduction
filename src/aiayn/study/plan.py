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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

from aiayn.inference.decoding import DecodeConfig
from aiayn.inference.evaluate import eval_tag
from aiayn.study.config import StudyConfig, Variant
from aiayn.study.storage import RunState
from aiayn.training.experiment import load_experiment

__all__ = [
    "DEFAULT",
    "EvalSpec",
    "PlannedRun",
    "Task",
    "all_run_names",
    "checkpoint_evals",
    "decoding_edges",
    "default_test",
    "default_valid",
    "experiment_name",
    "grid_evals",
    "is_alias",
    "main_evals",
    "main_run",
    "planned_runs",
    "select_checkpoint",
    "select_decoding",
    "select_training",
    "sensitivity_variants",
    "tasks",
    "training_edges",
    "variant_evals",
    "variant_run",
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
                if not variant.applies_to(model) or is_alias(study, model, variant):
                    continue
                runs += [
                    variant_run(study, model, group_name, variant, s)  # type: ignore[arg-type]
                    for s in group.seeds
                ]
    return runs


def sensitivity_variants(study: StudyConfig, model: str) -> list[Variant]:
    """Variants of the sensitivity grid that apply to ``model`` and change its config."""
    if model not in study.sensitivity.models:
        return []
    return [
        v
        for v in study.sensitivity.variants
        if v.applies_to(model) and not is_alias(study, model, v)
    ]


def all_run_names(study: StudyConfig) -> list[str]:
    """Every run name the study may use, including potential tuned seeds."""
    names = [r.name for r in planned_runs(study)]
    for model in study.sensitivity.models:
        for variant in sensitivity_variants(study, model):
            names += [variant_run(study, model, "tuned", variant, s).name for s in study.main_seeds]
    return sorted(set(names))


# ----------------------------------------------------------------------------- evaluations
def default_test(study: StudyConfig, checkpoint: str = "averaged") -> EvalSpec:
    return EvalSpec("test", checkpoint, study.default_beam, study.default_alpha)


def default_valid(study: StudyConfig, checkpoint: str = "averaged") -> EvalSpec:
    return EvalSpec(study.decoding.split, checkpoint, study.default_beam, study.default_alpha)


def main_evals(study: StudyConfig) -> tuple[EvalSpec, ...]:
    specs: tuple[EvalSpec, ...] = (
        default_test(study, "averaged"),
        default_test(study, "best"),
        EvalSpec("test", "best", 1, study.default_alpha),
    )
    if study.training_selection == "valid_bleu":
        specs += (default_valid(study),)
    return specs


def variant_evals(study: StudyConfig) -> tuple[EvalSpec, ...]:
    if study.training_selection == "valid_bleu":
        return (default_test(study), default_valid(study))
    return (default_test(study),)


def checkpoint_evals(study: StudyConfig) -> tuple[EvalSpec, ...]:
    return (default_valid(study, "averaged"), default_valid(study, "best"))


def grid_evals(study: StudyConfig, checkpoint: str = "averaged") -> tuple[EvalSpec, ...]:
    d = study.decoding
    specs = {}
    for beam in d.beams:
        for alpha in d.length_penalties if beam > 1 else (study.default_alpha,):
            spec = EvalSpec(d.split, checkpoint, beam, alpha)
            specs[spec.tag] = spec  # greedy appears once
    return tuple(specs.values())


# ----------------------------------------------------------------------------- selection
def _state(states: Mapping[str, RunState], name: str) -> RunState | None:
    return states.get(name)


def _mean_valid_bleu(
    study: StudyConfig, model: str, states: Mapping[str, RunState], tags: Sequence[str]
) -> dict[str, float] | None:
    """Mean BLEU per tag over the decoding seeds' main runs; None until all present."""
    totals = dict.fromkeys(tags, 0.0)
    seen = 0
    for seed in study.decoding.seeds:
        state = _state(states, main_run(study, model, seed).name)
        if state is None or state.failed:
            continue  # failed seeds do not block a selection
        if not state.completed:
            return None
        for tag in tags:
            if tag not in state.evals:
                return None
            totals[tag] += float(state.evals[tag]["bleu"])
        seen += 1
    if not seen:
        return None
    return {tag: total / seen for tag, total in totals.items()}


def select_checkpoint(study: StudyConfig, model: str, states: Mapping[str, RunState]) -> str | None:
    """``averaged`` or ``best`` by mean valid BLEU (ties keep ``averaged``); None until known."""
    if not study.select_checkpoint:
        return "averaged"
    averaged, best = (s.tag for s in checkpoint_evals(study))
    means = _mean_valid_bleu(study, model, states, (averaged, best))
    if means is None:
        return None
    return "best" if means[best] > means[averaged] + 1e-9 else "averaged"


def select_decoding(
    study: StudyConfig, model: str, states: Mapping[str, RunState]
) -> EvalSpec | None:
    """Valid-best (beam, alpha) on the selected checkpoint, as a test ``EvalSpec``.

    Ties keep the protocol default, then the smaller beam. None until complete.
    """
    checkpoint = select_checkpoint(study, model, states)
    if checkpoint is None:
        return None
    grid = grid_evals(study, checkpoint)
    means = _mean_valid_bleu(study, model, states, [s.tag for s in grid])
    if means is None:
        return None
    default_tag = default_valid(study, checkpoint).tag
    best = sorted(
        grid, key=lambda s: (-round(means[s.tag], 9), s.tag != default_tag, s.beam, s.alpha)
    )[0]
    return EvalSpec("test", checkpoint, best.beam, best.alpha)


def decoding_edges(study: StudyConfig, choice: EvalSpec) -> list[str]:
    """Selected decoding values at the edge of the grid (the optimum may lie outside)."""
    edges = []
    d = study.decoding
    if len(d.beams) > 1 and choice.beam == max(d.beams):
        edges.append("beam at grid maximum")
    if choice.beam > 1 and len(d.length_penalties) > 1:
        if choice.alpha == max(d.length_penalties):
            edges.append("alpha at grid maximum")
        elif choice.alpha == min(d.length_penalties):
            edges.append("alpha at grid minimum")
    return edges


def _training_score(study: StudyConfig, state: RunState | None) -> tuple[float, float] | None:
    """(primary, tie-break), both minimized; None while unfinished or unevaluated."""
    if state is None or not (state.completed or state.failed):
        return None
    nll = float("inf") if state.failed or state.best_valid_nll is None else state.best_valid_nll
    if study.training_selection == "valid_nll":
        return nll, 0.0
    if state.failed:
        return float("inf"), float("inf")
    result = state.evals.get(default_valid(study).tag)
    if result is None:
        return None
    return -float(result["bleu"]), nll


def select_training(study: StudyConfig, model: str, states: Mapping[str, RunState]) -> str | None:
    """Variant (or ``default``) with the best mean validation score over the sensitivity
    seeds: BLEU with NLL as tie-break, or NLL alone (Week 8). Failed runs rank last.
    None until every candidate run has finished and been evaluated."""
    group = study.sensitivity
    if model not in group.models:
        return DEFAULT
    candidates: dict[str, list[str]] = {
        DEFAULT: [main_run(study, model, s).name for s in group.seeds]
    }
    for variant in sensitivity_variants(study, model):
        candidates[variant.name] = [
            variant_run(study, model, "sensitivity", variant, s).name for s in group.seeds
        ]
    means: dict[str, tuple[float, float]] = {}
    for variant_name, names in candidates.items():
        scores = [_training_score(study, _state(states, n)) for n in names]
        known = [sc for sc in scores if sc is not None]
        if len(known) != len(scores):
            return None
        means[variant_name] = (
            round(sum(sc[0] for sc in known) / len(known), 9),
            round(sum(sc[1] for sc in known) / len(known), 9),
        )
    return min(means, key=lambda v: (means[v], v != DEFAULT))


def _numeric(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _lookup(tree: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(tree, dict) or part not in tree:
            return None
        tree = tree[part]
    return tree


def training_edges(study: StudyConfig, model: str, choice: str) -> list[str]:
    """Overrides of the selected variant whose value is the grid's minimum or maximum.

    A key's grid = the model's default value plus every applicable variant's value.
    """
    if choice == DEFAULT:
        return []
    variants = sensitivity_variants(study, model)
    chosen = next(v for v in variants if v.name == choice)
    defaults = load_experiment(study.models[model]).to_dict()
    edges = []
    for item in chosen.overrides:
        key, _, value = item.partition("=")
        picked = _numeric(value)
        if picked is None:
            continue
        values = {picked}
        base = _lookup(defaults, key)
        if isinstance(base, (int, float)) and not isinstance(base, bool):
            values.add(float(base))
        for v in variants:
            for other in v.overrides:
                k, _, val = other.partition("=")
                number = _numeric(val)
                if k == key and number is not None:
                    values.add(number)
        if len(values) >= 3 and picked in (max(values), min(values)):
            edges.append(f"{key} at grid {'maximum' if picked == max(values) else 'minimum'}")
    return edges


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

    def usable_main(model: str, seeds: Sequence[int]) -> list[PlannedRun]:
        out = []
        for seed in seeds:
            run = main_run(study, model, seed)
            state = _state(states, run.name)
            if state is not None and state.usable:
                out.append(run)
        return out

    if study.select_checkpoint:
        for model in study.models:
            for run in usable_main(model, study.decoding.seeds):
                todo.append(_task(run, states, checkpoint_evals(study), "checkpoint"))
    for model in study.models:
        checkpoint = select_checkpoint(study, model, states)
        if checkpoint is None:
            continue
        for run in usable_main(model, study.decoding.seeds):
            todo.append(_task(run, states, grid_evals(study, checkpoint), "decoding"))
    for model in study.models:
        selected = select_decoding(study, model, states)
        if selected is None:
            continue
        for run in usable_main(model, study.main_seeds):
            todo.append(_task(run, states, (selected,), "decoding_selected"))
    static = planned_runs(study)
    for group in ("ablation", "sensitivity"):
        for run in static:
            if run.group == group:
                todo.append(_task(run, states, variant_evals(study), group))
    for model in study.sensitivity.models:
        choice = select_training(study, model, states)
        if choice in (None, DEFAULT):
            continue
        variant = next(v for v in sensitivity_variants(study, model) if v.name == choice)
        for seed in study.main_seeds:
            if seed in study.sensitivity.seeds:
                continue
            run = variant_run(study, model, "tuned", variant, seed)
            todo.append(_task(run, states, (default_test(study),), "tuned"))
    return [t for t in todo if t is not None]
