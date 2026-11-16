"""Trainer behaviour on CPU: exact resumption, accumulation, budgets, guards, convergence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from aiayn.data.collate import Collator
from aiayn.data.config import load_data_config
from aiayn.data.loading import build_dataset
from aiayn.data.special_tokens import PAD_ID
from aiayn.models import build_transformer
from aiayn.training import Trainer, label_smoothed_nll
from aiayn.training.experiment import Experiment, experiment_from_dict
from aiayn.utils.config import ConfigError, config_to_dict
from tests.conftest import REPO_ROOT, small_model_config

pytestmark = pytest.mark.integration


def make_experiment(**train: Any) -> Experiment:
    data = config_to_dict(load_data_config(REPO_ROOT / "configs" / "data" / "copy_task.yaml"))
    data["copy"].update(num_train=96, num_valid=16, max_len=10)
    data["batching"]["max_tokens"] = 96
    model = config_to_dict(small_model_config(dropout=0.1, max_positions=64))
    settings: dict[str, Any] = {
        "name": "test",
        "seed": 3,
        "max_updates": 12,
        "update_freq": 2,
        "amp": "fp32",
        "device": "cpu",
        "schedule": {"kind": "inverse_sqrt", "warmup_steps": 4, "peak_lr": 1e-3},
        "log_every": 1,
        "validate_every": 5,
        "save_every": 4,
        "average_last": 2,
        "loggers": ["jsonl"],
    }
    settings.update(train)
    return experiment_from_dict({"name": "test", "data": data, "model": model, "train": settings})


def train_losses(run_dir: Path) -> list[tuple[int, float]]:
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [(r["step"], r["loss"]) for r in map(json.loads, lines) if r["split"] == "train"]


def final_weights(run_dir: Path) -> dict[str, torch.Tensor]:
    state = torch.load(run_dir / "checkpoints" / "last.pt", weights_only=True)
    return state["model"]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("reference") / "run"
    result = Trainer.create(make_experiment(), run_dir).train()
    assert result.status == "completed" and result.updates == 12
    return run_dir


def assert_identical_runs(a: Path, b: Path) -> None:
    assert train_losses(a) == train_losses(b)  # every update's loss, bit for bit
    wa, wb = final_weights(a), final_weights(b)
    assert wa.keys() == wb.keys()
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_stop_and_resume_is_bit_identical(reference_run: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = Trainer.create(make_experiment(max_updates=7), run_dir).train()  # stops mid-epoch
    assert first.status == "completed" and first.updates == 7
    second = Trainer.resume(run_dir, ["train.max_updates=12"]).train()
    assert second.updates == 12
    assert_identical_runs(reference_run, run_dir)


def test_budget_interruptions_after_every_update_are_bit_identical(
    reference_run: Path, tmp_path: Path
) -> None:
    """A zero-ish session budget stops after every single update (like a killed session)."""
    run_dir = tmp_path / "run"
    result = Trainer.create(make_experiment(time_budget_minutes=1e-9), run_dir).train()
    sessions = 1
    while result.status != "completed":
        assert result.status == "budget_exhausted"
        result = Trainer.resume(run_dir).train()
        sessions += 1
    assert sessions >= 12
    assert_identical_runs(reference_run, run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text("utf-8"))
    assert len(meta["sessions"]) == sessions
    assert meta["sessions"][-1]["status"] == "completed"


def test_run_directory_layout(reference_run: Path) -> None:
    for name in (
        "config.yaml",
        "metadata.json",
        "metrics.jsonl",
        "checkpoints/last.pt",
        "checkpoints/last.prev.pt",
        "checkpoints/best.pt",
    ):
        assert (reference_run / name).is_file(), name
    snapshots = sorted(p.name for p in (reference_run / "checkpoints" / "avg").glob("*.pt"))
    assert snapshots == ["step_00000008.pt", "step_00000012.pt"]  # average_last = 2
    lines = (reference_run / "metrics.jsonl").read_text("utf-8").splitlines()
    assert any(json.loads(line)["split"] == "valid" for line in lines)


def test_resume_refuses_a_changed_experiment(reference_run: Path) -> None:
    with pytest.raises(ConfigError, match="config hash"):
        Trainer.resume(reference_run, ["train.label_smoothing=0.0"])


def test_create_refuses_an_existing_run(reference_run: Path) -> None:
    with pytest.raises(FileExistsError):
        Trainer.create(make_experiment(), reference_run)


def test_resume_after_completion_is_a_no_op(reference_run: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(reference_run, copy)
    result = Trainer.resume(copy).train()
    assert result.status == "completed" and result.updates == 12


def test_gradient_accumulation_equals_one_large_batch() -> None:
    """ADR-017 §2: per-update token normalization makes accumulation exact."""
    data_cfg = load_data_config(REPO_ROOT / "configs" / "data" / "copy_task.yaml")
    dataset = build_dataset(data_cfg, "train")
    examples = [dataset[i] for i in range(6)]
    collate = Collator(PAD_ID)
    model = build_transformer(small_model_config(), vocab_size=20, pad_id=PAD_ID, seed=0).double()

    def grads(batches: list[Any]) -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        total = sum(b.num_tgt_tokens for b in batches)
        for b in batches:
            out = label_smoothed_nll(
                model(b.src, b.tgt_in), b.tgt_out, epsilon=0.1, ignore_index=PAD_ID
            )
            (out.loss_sum / total).backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    accumulated = grads([collate(examples[:2]), collate(examples[2:])])
    single = grads([collate(examples)])
    for a, b in zip(accumulated, single, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-9)


def test_overfitting_one_batch_without_smoothing_drives_loss_to_zero() -> None:
    """Spec amendment A2: overfit tests run with eps = 0 (the smoothed loss has a floor)."""
    data_cfg = load_data_config(REPO_ROOT / "configs" / "data" / "copy_task.yaml")
    batch = Collator(PAD_ID)([build_dataset(data_cfg, "train")[i] for i in range(8)])
    model = build_transformer(small_model_config(), vocab_size=20, pad_id=PAD_ID, seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.98), eps=1e-9)
    for _ in range(120):
        out = label_smoothed_nll(
            model(batch.src, batch.tgt_in), batch.tgt_out, epsilon=0.0, ignore_index=PAD_ID
        )
        loss = out.loss_sum / batch.num_tgt_tokens
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.02


def test_nonfinite_gradients_stop_training_and_dump_the_batch(tmp_path: Path) -> None:
    trainer = Trainer.create(make_experiment(), tmp_path / "run")
    with torch.no_grad():
        trainer.model.src_embedding.weight[5:].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train()
    assert list((tmp_path / "run").glob("nonfinite_update*.pt"))


@pytest.mark.slow
def test_copy_task_converges(tmp_path: Path) -> None:
    """Week 5 exit criterion (scaled down): teacher-forced accuracy on held-out copies."""
    experiment = experiment_from_dict(
        config_to_dict_experiment("copy_task_small"),
        [
            "train.max_updates=700",
            "train.loggers=[jsonl]",
            "train.validate_every=350",
            "train.device=cpu",
            "train.amp=fp32",
        ],
    )
    result = Trainer.create(experiment, tmp_path / "run").train()
    assert result.last_valid is not None and result.last_valid["accuracy"] > 0.97


def config_to_dict_experiment(name: str) -> dict[str, Any]:
    from aiayn.training.experiment import load_experiment

    return load_experiment(REPO_ROOT / "configs" / "experiment" / f"{name}.yaml").to_dict()
