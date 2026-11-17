"""Study config validation, plan expansion, aliases and the validation-only selection
rules (ADR-022), on synthetic run states."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from aiayn.study import RunState, load_study, planned_runs, tasks
from aiayn.study.plan import (
    DEFAULT,
    all_run_names,
    default_test,
    grid_evals,
    is_alias,
    main_evals,
    main_run,
    select_decoding,
    select_training,
    variant_run,
)
from aiayn.utils.config import ConfigError, load_yaml

REPO = Path(__file__).resolve().parents[2]
WEEK8 = REPO / "configs" / "study" / "week8.yaml"


@pytest.fixture(scope="module")
def study() -> Any:
    return load_study(WEEK8)


def state(
    *,
    completed: bool = True,
    failed: bool = False,
    nll: float | None = 2.0,
    evals: dict[str, float] | None = None,
) -> RunState:
    return RunState(
        path=Path("unused"),
        completed=completed,
        failed=failed,
        has_checkpoints=not completed,
        best_valid_nll=nll,
        parameters=1,
        evals={tag: {"bleu": bleu} for tag, bleu in (evals or {}).items()},
    )


def write_study(tmp_path: Path, **changes: Any) -> Path:
    import yaml

    raw = load_yaml(WEEK8)
    raw["models"] = {k: str(REPO / v) for k, v in raw["models"].items()}
    raw.update(changes)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- config
def test_week8_plan_shape(study: Any) -> None:
    runs = planned_runs(study)
    assert Counter(r.group for r in runs) == {"main": 20, "ablation": 30, "sensitivity": 60}
    assert len({r.name for r in runs}) == len(runs)
    # Week 6/7 runs are reused: the seed-1 defaults keep their names.
    assert {main_run(study, m, 1).name for m in study.models} == {
        "multi30k_tiny_s1",
        "multi30k_bilstm_luong_s1",
        "multi30k_gru_bahdanau_s1",
        "multi30k_lstm_seq2seq_s1",
    }
    assert all(r.overrides[-1] == f"train.seed={r.seed}" for r in runs)


def test_variants_equal_to_a_default_are_aliases(study: Any) -> None:
    by_name = {v.name: v for v in study.sensitivity.variants}
    assert is_alias(study, "transformer", by_name["dropout0.1"])
    assert not is_alias(study, "transformer", by_name["dropout0.3"])
    assert is_alias(study, "bilstm_luong", by_name["dropout0.3"])
    names = {r.name for r in planned_runs(study)}
    assert "multi30k_tiny-dropout0.1_s1" not in names and "multi30k_tiny-dropout0.3_s1" in names


def test_decoding_grid_tags(study: Any) -> None:
    tags = [s.tag for s in grid_evals(study)]
    assert len(tags) == 11 and len(set(tags)) == 11  # greedy once + 2 beams x 5 alphas
    assert "valid.averaged.greedy" in tags and "valid.averaged.beam4" in tags
    assert "valid.averaged.beam8-a1.5" in tags
    assert [s.tag for s in main_evals(study)] == [
        "test.averaged.beam4",
        "test.best.beam4",
        "test.best.greedy",
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"decoding": {"split": "test", "seeds": [1], "beams": [4], "length_penalties": [0.6]}},
            "never test",
        ),
        ({"decoding": {"seeds": [9], "beams": [4], "length_penalties": [0.6]}}, "subset"),
        ({"reference": "nope"}, "reference"),
        (
            {"sensitivity": {"seeds": [1], "variants": {"default": ["model.dropout=0.2"]}}},
            "invalid variant",
        ),
        ({"sensitivity": {"seeds": [1], "variants": {"x": ["train.seed=4"]}}}, "seeds come from"),
        (
            {"ablations": {"models": ["ghost"], "seeds": [1], "variants": {"x": []}}},
            "unknown model",
        ),
        ({"main_seeds": [1, 1]}, "duplicate"),
        ({"extra": 1}, "unknown keys"),
    ],
)
def test_config_validation(tmp_path: Path, changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_study(write_study(tmp_path, **changes))


# ----------------------------------------------------------------------------- selection
def grid_states(
    study: Any, model: str, best_tag: str, *, missing: str | None = None
) -> dict[str, RunState]:
    states = {}
    for seed in study.decoding.seeds:
        evals = {s.tag: (40.0 if s.tag == best_tag else 30.0) for s in grid_evals(study)}
        if missing and seed == study.decoding.seeds[-1]:
            evals.pop(missing)
        states[main_run(study, model, seed).name] = state(evals=evals)
    return states


def test_decoding_selection_uses_valid_and_waits_for_the_grid(study: Any) -> None:
    states = grid_states(study, "gru_bahdanau", "valid.averaged.beam8-a1")
    chosen = select_decoding(study, "gru_bahdanau", states)
    assert chosen is not None and (chosen.split, chosen.beam, chosen.alpha) == ("test", 8, 1.0)
    incomplete = grid_states(
        study, "gru_bahdanau", "valid.averaged.beam8-a1", missing="valid.averaged.beam4"
    )
    assert select_decoding(study, "gru_bahdanau", incomplete) is None


def test_decoding_ties_keep_the_protocol_default(study: Any) -> None:
    states = {
        main_run(study, "transformer", s).name: state(
            evals={g.tag: 30.0 for g in grid_evals(study)}
        )
        for s in study.decoding.seeds
    }
    chosen = select_decoding(study, "transformer", states)
    assert chosen is not None and chosen.tag == default_test(study).tag


def sensitivity_states(study: Any, model: str, nll: dict[str, float]) -> dict[str, RunState]:
    states = {}
    for seed in study.sensitivity.seeds:
        states[main_run(study, model, seed).name] = state(nll=nll.get(DEFAULT, 2.0))
        for v in study.sensitivity.variants:
            if not is_alias(study, model, v):
                name = variant_run(study, model, "sensitivity", v, seed).name
                states[name] = state(nll=nll.get(v.name, 2.5))
    return states


def test_training_selection_by_mean_validation_nll(study: Any) -> None:
    states = sensitivity_states(study, "transformer", {"dropout0.3": 1.8})
    assert select_training(study, "transformer", states) == "dropout0.3"
    states = sensitivity_states(study, "transformer", {})
    assert select_training(study, "transformer", states) == DEFAULT
    # A diverged variant counts as +inf, never as selected.
    diverged = sensitivity_states(study, "transformer", {"lr_double": 0.1})
    for seed in study.sensitivity.seeds:
        name = variant_run(
            study, "transformer", "sensitivity", study.sensitivity.variants[1], seed
        ).name
        diverged[name] = state(completed=False, failed=True, nll=None)
    assert select_training(study, "transformer", diverged) == DEFAULT


def test_training_selection_waits_for_unfinished_runs(study: Any) -> None:
    states = sensitivity_states(study, "transformer", {"dropout0.3": 1.8})
    first = variant_run(study, "transformer", "sensitivity", study.sensitivity.variants[0], 1).name
    states[first] = state(completed=False, nll=None)
    assert select_training(study, "transformer", states) is None


# ----------------------------------------------------------------------------- tasks
def test_initial_tasks_are_prioritized(study: Any) -> None:
    todo = tasks(study, {})
    phases = [t.phase for t in todo]
    assert phases == sorted(phases, key=["main", "ablation", "sensitivity"].index)
    assert all(t.train for t in todo) and len(todo) == 110


def test_completed_runs_produce_follow_up_tasks(study: Any) -> None:
    states: dict[str, RunState] = {}
    for model in study.models:
        for seed in study.main_seeds:
            states[main_run(study, model, seed).name] = state(
                evals={s.tag: 30.0 for s in main_evals(study)}
            )
    todo = tasks(study, states)
    assert Counter(t.phase for t in todo)["decoding"] == 4 * len(study.decoding.seeds)
    assert not any(t.phase == "main" for t in todo)
    # Grid done with a non-default winner -> test evaluations at that setting for every seed.
    for model in study.models:
        for name, s in grid_states(study, model, "valid.averaged.beam8").items():
            states[name] = state(
                evals={**states[name].evals, **{k: v["bleu"] for k, v in s.evals.items()}}
            )
    todo = tasks(study, states)
    selected = [t for t in todo if t.phase == "decoding_selected"]
    assert len(selected) == 4 * len(study.main_seeds)
    assert {t.evals[0].tag for t in selected} == {"test.averaged.beam8"}


def test_tuned_runs_appear_after_selection_and_failed_runs_are_skipped(study: Any) -> None:
    states = sensitivity_states(study, "transformer", {"lr_half": 1.5})
    names = [t.run.name for t in tasks(study, states) if t.phase == "tuned"]
    assert names == ["multi30k_tiny-lr_half_s4", "multi30k_tiny-lr_half_s5"]
    assert set(all_run_names(study)) >= set(names)
    failed = main_run(study, "transformer", 2).name
    states[failed] = state(completed=False, failed=True, nll=None)
    assert failed not in [t.run.name for t in tasks(study, states)]
