"""Study execution (ADR-022): the runner end to end with fake training/evaluation
(budget stops, continuation, failures, adoption of input runs), slimming of a really
trained run, and the report."""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from sacrebleu.metrics.bleu import BLEU

from aiayn.inference import DecodeConfig, evaluate_run
from aiayn.study import StudyEnv, collect_states, load_study, read_state, run_study, slim_run, tasks
from aiayn.study.analysis import build_study_report
from aiayn.study.plan import EvalSpec, PlannedRun, main_run
from aiayn.study.runner import default_train
from aiayn.training import Trainer
from tests.integration.test_trainer import make_experiment

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
WORDS = [
    "a",
    "man",
    "woman",
    "dog",
    "runs",
    "sits",
    "on",
    "the",
    "red",
    "blue",
    "street",
    "park",
    "ball",
]
REFS = [" ".join(random.Random(i).choice(WORDS) for _ in range(6 + i % 5)) for i in range(60)]
QUALITY = {"transformer": 0.8, "gru_bahdanau": 0.6}
NLL = {"default": 2.0, "lr_half": 2.2, "dropout0.3": 1.9}  # dropout0.3 wins for the Transformer


def small_study(tmp_path: Path) -> Any:
    raw = {
        "name": "mini",
        "reference": "transformer",
        "models": {
            "transformer": str(REPO / "configs/experiment/multi30k_tiny.yaml"),
            "gru_bahdanau": str(REPO / "configs/experiment/multi30k_gru_bahdanau.yaml"),
        },
        "main_seeds": [1, 2, 3],
        "bootstrap_samples": 100,
        "decoding": {"seeds": [1, 2], "beams": [1, 4], "length_penalties": [0.6, 1.0]},
        "sensitivity": {
            "seeds": [1, 2],
            "variants": {
                "lr_half": ["train.schedule.peak_lr=3.5e-4"],
                "dropout0.3": ["model.dropout=0.3"],
            },
        },
        "ablations": {
            "models": ["transformer"],
            "seeds": [1, 2],
            "variants": {"heads8": ["model.num_heads=8"]},
        },
    }
    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_study(path)


class FakeTrainer:
    def __init__(self, fail: set[str] | None = None, exit_codes: dict[str, int] | None = None):
        self.calls: list[str] = []
        self.fail = fail or set()
        self.exit_codes = exit_codes or {}

    def __call__(self, run: PlannedRun, run_dir: Path, resume: bool, budget: int) -> int:
        self.calls.append(run.name)
        if run.name in self.exit_codes:
            return self.exit_codes.pop(run.name)
        if run.name in self.fail:
            return 1
        (run_dir / "checkpoints" / "avg").mkdir(parents=True, exist_ok=True)
        weights = {"w": torch.full((2,), float(run.seed))}
        for step in (10, 20):
            torch.save(
                {"w": weights["w"] + step}, run_dir / "checkpoints" / "avg" / f"step_{step:08d}.pt"
            )
        torch.save(
            {"model": weights, "optimizer": {"step": torch.tensor(1)}},
            run_dir / "checkpoints" / "best.pt",
        )
        torch.save({"model": weights}, run_dir / "checkpoints" / "last.pt")
        config = {
            "name": run.model,
            "data": {"corpus": "same"},
            "model": {"name": run.model},
            "train": {"seed": run.seed},
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        nll = (
            NLL.get(run.variant, 2.1)
            + (0.0 if run.model == "transformer" else 0.3)
            + run.seed * 0.01
        )
        meta = {
            "parameters": 1000,
            "tokenizer_fingerprint": "fp",
            "best_valid_nll": nll,
            "sessions": [{"status": "completed", "start_update": 0, "end_update": 50}],
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        return 0


def fake_eval(run_dir: Path, spec: EvalSpec) -> None:
    model = "transformer" if "tiny" in run_dir.name else "gru_bahdanau"
    quality = QUALITY[model] + (
        0.1 if model == "gru_bahdanau" and spec.alpha == 1.0 and spec.beam > 1 else 0.0
    )
    rng = random.Random(zlib.crc32(f"{run_dir.name}{spec.split}{spec.checkpoint}".encode()))
    hyps = [r if rng.random() < quality else " ".join(r.split()[:2]) for r in REFS]
    score = BLEU().corpus_score(hyps, [REFS])
    out = run_dir / "eval"
    out.mkdir(exist_ok=True)
    (out / f"{spec.tag}.hyp").write_text("\n".join(hyps) + "\n", encoding="utf-8")
    result = {
        "bleu": round(score.score, 2),
        "chrf": 50.0,
        "length_ratio": score.sys_len / score.ref_len,
    }
    (out / f"{spec.tag}.json").write_text(json.dumps(result), encoding="utf-8")


def test_full_study_runs_every_phase_and_slims(tmp_path: Path) -> None:
    study = small_study(tmp_path)
    env = StudyEnv(tmp_path / "work")
    trainer = FakeTrainer()
    outcome = run_study(study, env, minutes=1e6, train_fn=trainer, eval_fn=fake_eval)
    assert outcome.status == "complete" and outcome.remaining_tasks == 0
    assert len(trainer.calls) == len(set(trainer.calls))  # nothing trained twice
    # 6 main + 2 ablation + 2x2x2 sensitivity (dropout0.3 is the GRU default) - aliases + tuned
    states = collect_states(study, env)
    assert tasks(study, states) == []
    assert "multi30k_tiny-dropout0.3_s3" in trainer.calls  # tuned: selected on validation NLL
    assert "multi30k_gru_bahdanau-dropout0.3_s1" not in trainer.calls  # alias of the default
    # Decoding grid on valid picked beam 4, alpha 1.0 for the GRU -> test on every seed.
    for seed in (1, 2, 3):
        assert "test.averaged.beam4-a1" in states[main_run(study, "gru_bahdanau", seed).name].evals
        assert (
            "test.averaged.beam4-a1" not in states[main_run(study, "transformer", seed).name].evals
        )
    main_dir = env.work_root / "multi30k_tiny_s1" / "checkpoints"
    assert sorted(p.name for p in main_dir.iterdir()) == ["averaged.pt", "best.pt"]
    assert set(torch.load(main_dir / "best.pt", weights_only=True)) == {"w"}  # weights only
    variant_dir = env.work_root / "multi30k_tiny-heads8_s1" / "checkpoints"
    assert sorted(p.name for p in variant_dir.iterdir()) == ["averaged.pt"]
    assert torch.equal(
        torch.load(variant_dir / "averaged.pt", weights_only=True)["w"], torch.full((2,), 16.0)
    )


def test_budget_stops_and_the_next_session_continues(tmp_path: Path) -> None:
    study = small_study(tmp_path)
    env = StudyEnv(tmp_path / "work")
    trainer = FakeTrainer()
    now = [0.0]

    def clock() -> float:
        now[0] += 60.0  # every check costs a minute
        return now[0]

    first = run_study(
        study, env, minutes=40, reserve_minutes=0, train_fn=trainer, eval_fn=fake_eval, clock=clock
    )
    assert first.status == "budget" and first.remaining_tasks > 0
    partial_report, data = build_study_report(study, collect_states(study, env), REFS)
    assert "partial results" in partial_report and data["status"]["pending_tasks"] > 0
    second = run_study(study, env, minutes=1e6, train_fn=trainer, eval_fn=fake_eval)
    assert second.status == "complete"
    assert len(trainer.calls) == len(set(trainer.calls))


def test_resumable_exit_ends_the_session(tmp_path: Path) -> None:
    study = small_study(tmp_path)
    trainer = FakeTrainer(exit_codes={"multi30k_tiny_s1": 3})
    outcome = run_study(
        study, StudyEnv(tmp_path / "work"), minutes=1e6, train_fn=trainer, eval_fn=fake_eval
    )
    assert outcome.status == "budget" and trainer.calls == ["multi30k_tiny_s1"]


def test_failed_runs_are_recorded_and_reported(tmp_path: Path) -> None:
    study = small_study(tmp_path)
    env = StudyEnv(tmp_path / "work")
    failing = "multi30k_tiny-lr_half_s2"
    outcome = run_study(
        study, env, minutes=1e6, train_fn=FakeTrainer(fail={failing}), eval_fn=fake_eval
    )
    assert outcome.status == "complete" and outcome.failed == [failing]
    assert read_state(env.work_root / failing).failed
    report, data = build_study_report(study, collect_states(study, env), REFS)
    assert failing in data["status"]["failed"] and "Failed runs" in report


def test_runs_from_inputs_are_reused_and_adopted(tmp_path: Path) -> None:
    study = small_study(tmp_path)
    previous = StudyEnv(tmp_path / "session1" / "runs")
    run_study(study, previous, minutes=1e6, train_fn=FakeTrainer(), eval_fn=fake_eval)
    env = StudyEnv(tmp_path / "session2", input_roots=[tmp_path / "session1"])
    trainer = FakeTrainer()
    outcome = run_study(study, env, minutes=1e6, train_fn=trainer, eval_fn=fake_eval)
    assert outcome.status == "complete" and trainer.calls == []
    adopted = sorted(p.name for p in env.work_root.iterdir())
    assert adopted == sorted(p.name for p in previous.work_root.iterdir())  # full snapshot


def test_report_sections_and_selection(tmp_path: Path) -> None:
    study = small_study(tmp_path)
    env = StudyEnv(tmp_path / "work")
    run_study(study, env, minutes=1e6, train_fn=FakeTrainer(), eval_fn=fake_eval)
    report, data = build_study_report(study, collect_states(study, env), REFS)
    for heading in (
        "## 1. Main comparison",
        "## 2. Comparison with decoding selected on validation",
        "## 3. Comparison with training configuration selected on validation",
        "## 4. Training sensitivity",
        "## 5. Decoding sensitivity",
        "## 6. Checkpoint averaging and beam search",
        "## 7. Ablations",
        "## 8. Statistical notes and checks",
    ):
        assert heading in report
    assert "— complete." in report
    assert data["training_selection"] == {"transformer": "dropout0.3", "gru_bahdanau": "default"}
    assert data["decoding_selection"]["gru_bahdanau"] == [4, 1.0]
    main = {e["label"]: e for e in data["main"]}
    assert len(main["transformer"]["bleu"]) == 3 and main["transformer"]["test"] is None
    gru = main["gru_bahdanau"]["test"]
    assert gru["delta"] < 0 and gru["ci"][0] <= gru["delta"] <= gru["ci"][1]
    assert "PASS: one tokenizer fingerprint" in report and "| ref |" in report
    assert next(iter(study.models)) == "transformer"


def test_default_train_builds_the_right_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = small_study(tmp_path)
    calls: list[list[str]] = []

    class Done:
        returncode = 0

    monkeypatch.setattr(
        "aiayn.study.runner.subprocess.run", lambda cmd, **_: calls.append(cmd) or Done()
    )
    train = default_train(StudyEnv(tmp_path / "work", repo_root=REPO, python="py"))
    run = main_run(study, "transformer", 2)
    assert train(run, tmp_path / "work" / run.name, False, 120) == 0
    assert train(run, tmp_path / "work" / run.name, True, 30) == 0
    new, resume = calls
    assert new[:3] == ["py", "-u", "scripts/train.py"] and "--experiment" in new
    assert "train.seed=2" in new and new[-1] == "train.time_budget_minutes=120"
    assert "--resume" in resume and "--experiment" not in resume and "train.seed=2" not in resume


def test_slimming_a_trained_run_keeps_evaluations_identical(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    Trainer.create(make_experiment(max_updates=8, save_every=2, average_last=3), run_dir).train()
    decode = DecodeConfig(beam_size=2, max_len_b=12)
    before = {
        c: evaluate_run(
            run_dir, split="valid", checkpoint=c, decode=decode, device="cpu", write=False
        )
        for c in ("averaged", "best")
    }
    freed = slim_run(run_dir, keep_best=True)
    assert freed > 0 and slim_run(run_dir, keep_best=True) == 0  # idempotent
    ckpt = run_dir / "checkpoints"
    assert sorted(p.name for p in ckpt.iterdir()) == ["averaged.pt", "best.pt"]
    assert json.loads((run_dir / "metadata.json").read_text("utf-8"))["slimmed"] is True
    for c, result in before.items():
        after = evaluate_run(
            run_dir, split="valid", checkpoint=c, decode=decode, device="cpu", write=False
        )
        assert after["bleu"] == result["bleu"] and after["chrf"] == result["chrf"]
    slim_run(run_dir, keep_best=False)
    assert sorted(p.name for p in ckpt.iterdir()) == ["averaged.pt"]


def test_incomplete_runs_are_never_slimmed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    Trainer.create(make_experiment(max_updates=4, time_budget_minutes=1e-9), run_dir).train()
    with pytest.raises(ValueError, match="completed"):
        slim_run(run_dir, keep_best=True)
