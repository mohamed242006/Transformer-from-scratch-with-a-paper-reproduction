"""Train briefly -> Translator.from_run -> evaluate_run, for the copy task and a synthetic
translation corpus (tokenizer, raw references, sacreBLEU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiayn.inference import DecodeConfig, Translator, evaluate_run
from aiayn.training import Trainer, ensure_data
from aiayn.training.experiment import experiment_from_dict
from aiayn.utils.config import config_to_dict
from tests.conftest import small_model_config
from tests.integration.test_trainer import make_experiment
from tests.unit.test_prepare import translation_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def copy_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("eval") / "copy"
    Trainer.create(make_experiment(max_updates=8, save_every=4), run_dir).train()
    return run_dir


@pytest.mark.parametrize("checkpoint", ["best", "last", "averaged"])
def test_copy_run_evaluation_writes_results(copy_run: Path, checkpoint: str) -> None:
    result = evaluate_run(
        copy_run,
        split="valid",
        checkpoint=checkpoint,
        decode=DecodeConfig(beam_size=2, max_len_b=12),
        device="cpu",
    )
    tag = f"valid.{checkpoint}.beam2"
    saved = json.loads((copy_run / "eval" / f"{tag}.json").read_text("utf-8"))
    assert saved["bleu"] == result["bleu"] and saved["sentences"] == 16
    lines = (copy_run / "eval" / f"{tag}.hyp").read_text("utf-8").splitlines()
    assert len(lines) == 16  # one line per source sentence, corpus order
    assert "bleu_signature" in saved and saved["bleu_by_length"]


def test_translator_preserves_input_order(copy_run: Path) -> None:
    translator = Translator.from_run(copy_run, device="cpu", max_tokens=40)  # many batches
    sources = [[5, 6, 7], [8], [9, 10, 11, 12, 13], [4, 4]]
    greedy = translator.translate_ids(sources, DecodeConfig(beam_size=1, max_len_b=8))
    singles = [
        translator.translate_ids([s], DecodeConfig(beam_size=1, max_len_b=8))[0] for s in sources
    ]
    assert [h.tokens for h in greedy] == [h.tokens for h in singles]
    with pytest.raises(ValueError, match="no tokenizer"):
        translator.translate(["text"])


def test_translation_run_end_to_end(corpus_dir: Path, tmp_path: Path) -> None:
    data = config_to_dict(translation_config(corpus_dir, tmp_path / "data"))
    raw = {
        "name": "synthetic",
        "data": data,
        "model": config_to_dict(small_model_config(max_positions=128)),
        "train": {
            "name": "t",
            "max_updates": 6,
            "amp": "fp32",
            "device": "cpu",
            "loggers": ["jsonl"],
            "save_every": 3,
            "schedule": {"kind": "inverse_sqrt", "warmup_steps": 2, "peak_lr": 1e-3},
        },
    }
    experiment = experiment_from_dict(raw)
    ensure_data(experiment)
    run_dir = tmp_path / "run"
    Trainer.create(experiment, run_dir).train()

    translator = Translator.from_run(run_dir, device="cpu")
    out = translator.translate(["the dog runs .", "a small cat sits ."], DecodeConfig(beam_size=2))
    assert len(out) == 2 and all(isinstance(s, str) for s in out)

    result = evaluate_run(run_dir, split="test", decode=DecodeConfig(beam_size=2), device="cpu")
    assert result["sentences"] == 60  # every raw test line, unfiltered
    assert 0.0 <= result["bleu"] <= 100.0 and "case:mixed" in result["bleu_signature"]
