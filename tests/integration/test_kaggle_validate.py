"""End-to-end validation on CPU: synthetic corpus (file:// "mirror"), tiny model config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aiayn.data.config import DataConfig
from aiayn.diagnostics.packaging import build_data_bundle
from aiayn.diagnostics.validation import ValidationOptions, run_validation
from aiayn.utils.config import config_to_dict
from tests.conftest import REPO_ROOT
from tests.unit.test_prepare import translation_config

pytestmark = pytest.mark.integration


def write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def small_model_yaml(tmp_path: Path) -> Path:
    return write_yaml(
        tmp_path / "model.yaml",
        {
            "name": "validate_small",
            "d_model": 32,
            "num_heads": 4,
            "d_ff": 64,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "max_positions": 128,
        },
    )


def data_yaml(tmp_path: Path, config: DataConfig) -> Path:
    raw = config_to_dict(config)
    raw["batching"]["max_tokens"] = 512
    return write_yaml(tmp_path / "data.yaml", raw)


def test_translation_validation_end_to_end(corpus_dir: Path, tmp_path: Path) -> None:
    config = translation_config(corpus_dir, tmp_path / "data")
    options = ValidationOptions(
        data_config=data_yaml(tmp_path, config),
        model_configs=[small_model_yaml(tmp_path)],
        train_tokenizer_if_missing=True,
        retrain_tokenizer_check=True,
        device="cpu",
        steps=2,
        warmup=1,
        check_internet=False,
    )
    report = run_validation(options, output_dir=tmp_path)
    statuses = {r.name: r.status for r in report.results}
    assert report.exit_code() == 0, report.format_table()
    for name in (
        "env.python",
        "deps.versions",
        "data.prepare",
        "tokenizer.load",
        "tokenizer.deterministic",
        "data.loader",
        "model[validate_small].build",
        "model[validate_small].forward",
        "model[validate_small].backward",
        "model[validate_small].throughput",
    ):
        assert statuses[name] == "pass", (name, report.format_table())
    assert "cuda.available" not in statuses  # device=cpu: CUDA checks are not attempted
    memory = report.get("model[validate_small].memory")
    assert memory is not None and memory.status == "skip"  # estimated, not measured, on CPU
    assert memory.details["estimated_peak_mib"] > 0
    data = json.loads(report.write_json(tmp_path / "report.json").read_text(encoding="utf-8"))
    assert data["ok"] is True


def test_offline_mirror_and_expected_fingerprint(corpus_dir: Path, tmp_path: Path) -> None:
    """Kaggle-style run: archives from a data bundle used as mirror, tokenizer from the bundle,
    fingerprint checked; a wrong fingerprint is a critical failure."""
    source = translation_config(corpus_dir, tmp_path / "local")
    run_validation(  # prepare + train the tokenizer locally, as on the dev machine
        ValidationOptions(
            data_config=data_yaml(tmp_path, source),
            model_configs=[],
            train_tokenizer_if_missing=True,
            device="cpu",
            check_internet=False,
        )
    )
    bundle = tmp_path / "bundle"
    build_data_bundle(source, bundle, kaggle_user="someone")
    fingerprint = (bundle / "tokenizer_fingerprint.txt").read_text(encoding="utf-8").strip()
    model_file = next(bundle.glob("*.model"))

    kaggle_dir = tmp_path / "kaggle"
    kaggle_dir.mkdir()
    options = ValidationOptions(
        data_config=data_yaml(kaggle_dir, source),
        model_configs=[small_model_yaml(kaggle_dir)],
        data_overrides=(f"translation.root={(tmp_path / 'kaggle_working').as_posix()}",),
        data_mirrors=[bundle],
        tokenizer_path=model_file,
        expected_fingerprint=fingerprint,
        device="cpu",
        steps=1,
        warmup=0,
        check_internet=False,
    )
    assert run_validation(options).exit_code() == 0
    options.expected_fingerprint = "0" * 64
    report = run_validation(options)
    fp = report.get("tokenizer.fingerprint")
    assert fp is not None and fp.status == "fail" and report.exit_code() == 1
    # A mismatch is critical but does not hide the model, VRAM and throughput results.
    assert report.passed("model[validate_small].forward")


def test_copy_task_validation_runs_without_tokenizer(tmp_path: Path) -> None:
    options = ValidationOptions(
        data_config=REPO_ROOT / "configs" / "data" / "copy_task.yaml",
        data_overrides=("copy.num_train=200", "copy.num_valid=40"),
        model_configs=[small_model_yaml(tmp_path)],
        device="cpu",
        steps=1,
        warmup=0,
        check_internet=False,
    )
    report = run_validation(options)
    assert report.exit_code() == 0, report.format_table()
    tok = report.get("tokenizer.load")
    assert tok is not None and tok.status == "skip"


def test_missing_tokenizer_without_training_is_critical(corpus_dir: Path, tmp_path: Path) -> None:
    config = translation_config(corpus_dir, tmp_path / "data")
    options = ValidationOptions(
        data_config=data_yaml(tmp_path, config),
        model_configs=[small_model_yaml(tmp_path)],
        device="cpu",
        check_internet=False,
    )
    report = run_validation(options)
    assert report.exit_code() == 1
    assert report.get("data.loader") is None  # later checks are not attempted


def test_training_resume_check_is_bit_identical_on_cpu(tmp_path: Path) -> None:
    options = ValidationOptions(
        data_config=REPO_ROOT / "configs" / "data" / "copy_task.yaml",
        data_overrides=("copy.num_train=64", "copy.num_valid=16"),
        model_configs=[],
        device="cpu",
        check_internet=False,
        check_training_resume=True,
    )
    report = run_validation(options)
    check = report.get("training.resume")
    assert check is not None and check.status == "pass", report.format_table()
    assert check.details["bit_identical"] is True
