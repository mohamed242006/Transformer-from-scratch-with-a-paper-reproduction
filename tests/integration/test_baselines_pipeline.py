"""RNN baselines through the *same* pipeline as the Transformer (Week 7 exit criterion):
Trainer (bit-identical resume) -> evaluate_run -> compare_runs with fairness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aiayn.inference import DecodeConfig, build_comparison, evaluate_run, load_references
from aiayn.inference.comparison import summarize_run
from aiayn.models import RNNSeq2Seq
from aiayn.training import Trainer
from aiayn.training.experiment import Experiment, experiment_from_dict
from tests.integration.test_trainer import assert_identical_runs, make_experiment

pytestmark = pytest.mark.integration

RNN_MODEL: dict[str, Any] = {
    "name": "test_rnn",
    "architecture": "rnn",
    "cell": "lstm",
    "attention": "luong",
    "embedding_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "bidirectional_encoder": True,
    "dropout": 0.1,
    "max_positions": 64,
}


def rnn_experiment(model: dict[str, Any] | None = None, **train: Any) -> Experiment:
    raw = make_experiment(**train).to_dict()
    raw["model"] = dict(model or RNN_MODEL)
    return experiment_from_dict(raw)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("baselines")
    out = {}
    for name, experiment in {
        "transformer": make_experiment(max_updates=8),
        "rnn": rnn_experiment(max_updates=8, clip_norm=1.0),
    }.items():
        run_dir = root / name
        result = Trainer.create(experiment, run_dir).train()
        assert result.status == "completed"
        evaluate_run(
            run_dir,
            split="valid",
            checkpoint="averaged",
            decode=DecodeConfig(beam_size=2, max_len_b=12),
            device="cpu",
        )
        out[name] = run_dir
    return out


@pytest.mark.parametrize(
    "variant",
    [
        {"cell": "gru", "attention": "bahdanau", "num_layers": 1},
        {"cell": "lstm", "attention": "none", "bidirectional_encoder": False},
    ],
)
def test_rnn_stop_and_resume_is_bit_identical(tmp_path: Path, variant: dict[str, Any]) -> None:
    model = {**RNN_MODEL, **variant}
    reference = tmp_path / "reference"
    Trainer.create(rnn_experiment(model, max_updates=8), reference).train()
    resumed = tmp_path / "resumed"
    Trainer.create(rnn_experiment(model, max_updates=5), resumed).train()  # mid-epoch stop
    assert Trainer.resume(resumed, ["train.max_updates=8"]).train().updates == 8
    assert_identical_runs(reference, resumed)


def test_trainer_builds_the_rnn_and_records_it(runs: dict[str, Path]) -> None:
    trainer = Trainer.resume(runs["rnn"])
    assert isinstance(trainer.model, RNNSeq2Seq)
    meta = json.loads((runs["rnn"] / "metadata.json").read_text("utf-8"))
    assert meta["parameters"] == sum(p.numel() for p in trainer.model.parameters())


def test_rnn_evaluation_uses_the_shared_pipeline(runs: dict[str, Path]) -> None:
    result = json.loads((runs["rnn"] / "eval" / "valid.averaged.beam2.json").read_text("utf-8"))
    assert result["sentences"] == 16 and result["decode"]["beam_size"] == 2
    hyps = (runs["rnn"] / "eval" / "valid.averaged.beam2.hyp").read_text("utf-8").splitlines()
    assert len(hyps) == len(load_references(runs["rnn"], "valid")) == 16


def test_comparison_reports_results_and_fairness(runs: dict[str, Path]) -> None:
    text = build_comparison(
        [runs["transformer"], runs["rnn"]], tag="valid.averaged.beam2", n_samples=50
    )
    assert "PASS: same data configuration" in text
    assert "PASS: same tokenizer fingerprint" in text and "PASS: same reference lines" in text
    assert "PASS: same decoding and checkpoint" in text
    assert "rnn: training differences: clip_norm: 0.0 -> 1.0" in text  # declared: passes
    assert "(undeclared" not in text
    # The toy models are not size-matched: the parameter check must catch it.
    assert "**FAIL**: rnn:" in text and "limit ±10%" in text
    assert "| `transformer` |" in text and "| `rnn` |" in text and "ref |" in text
    assert "paired bootstrap resampling (50 samples)" in text
    assert "Across seeds" not in text  # same experiment name, different models


def test_comparison_flags_undeclared_training_differences(
    runs: dict[str, Path], tmp_path: Path
) -> None:
    other = tmp_path / "other"
    Trainer.create(rnn_experiment(max_updates=8, label_smoothing=0.2), other).train()
    evaluate_run(
        other,
        split="valid",
        checkpoint="averaged",
        decode=DecodeConfig(beam_size=2, max_len_b=12),
        device="cpu",
    )
    text = build_comparison(
        [runs["transformer"], other], tag="valid.averaged.beam2", significance=False
    )
    assert "(undeclared: label_smoothing)" in text
    assert "Significance not computed" in text


def test_summary_requires_the_evaluation(runs: dict[str, Path]) -> None:
    summary = summarize_run(runs["rnn"], "valid.averaged.beam2")
    assert summary.description.startswith("BiLSTM 2L H=24 + Luong")
    with pytest.raises(FileNotFoundError, match="evaluate"):
        summarize_run(runs["rnn"], "test.best.beam4")
