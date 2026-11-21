"""The frozen Week 9 Phase 2 benchmark (ADR-023 Amendment 3): final hyperparameters,
9 main runs and no tuning variants."""

from __future__ import annotations

from pathlib import Path

from aiayn.inference.comparison import data_signature, update_tokens
from aiayn.study import load_study, planned_runs
from aiayn.training.experiment import load_experiment

REPO = Path(__file__).resolve().parents[2]


def test_final_hyperparameters_are_frozen() -> None:
    exp = REPO / "configs" / "experiment"
    t = load_experiment(exp / "iwslt14_transformer.yaml")
    luong = load_experiment(exp / "iwslt14_bilstm_luong.yaml")
    gru = load_experiment(exp / "iwslt14_gru_bahdanau.yaml")
    assert t.model.norm_position == "pre" and t.train.schedule.peak_lr == 0.001  # type: ignore[union-attr]
    assert luong.train.schedule.peak_lr == 0.0056 and luong.train.clip_norm == 1.0
    assert gru.train.schedule.peak_lr == 0.0028 and gru.train.update_freq == 2
    assert all(e.train.max_updates == 30000 for e in (t, luong, gru))
    configs = [e.to_dict() for e in (t, luong, gru)]
    assert {update_tokens(c) for c in configs} == {8192}
    assert len({data_signature(c) for c in configs}) == 1


def test_benchmark_study_is_three_models_by_three_seeds() -> None:
    study = load_study(REPO / "configs" / "study" / "week9_benchmark.yaml")
    runs = planned_runs(study)
    assert sorted(r.name for r in runs) == sorted(
        f"iwslt14_{m}_s{s}"
        for m in ("transformer", "bilstm_luong", "gru_bahdanau")
        for s in (1, 2, 3)
    )
    assert {r.group for r in runs} == {"main"}
    assert study.select_checkpoint and study.training_selection == "valid_bleu"
    assert study.decoding.split == "valid"
