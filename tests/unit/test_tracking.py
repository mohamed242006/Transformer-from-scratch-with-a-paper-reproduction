from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aiayn.tracking import JsonlLogger, MultiLogger, StdoutLogger, build_loggers


def test_jsonl_appends_one_flushed_record_per_call(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "m.jsonl")
    logger.log({"loss": 1.5, "lr": 1e-3}, step=10, split="train")
    lines = (tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()  # before close
    assert len(lines) == 1
    logger.log({"nll": float("nan")}, step=20, split="valid")
    logger.close()
    records = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text("utf-8").splitlines()]
    assert records[0]["split"] == "train" and records[0]["step"] == 10 and records[0]["loss"] == 1.5
    assert records[1]["nll"] is None  # non-finite values become null, keeping valid JSON


def test_jsonl_resumes_by_appending(tmp_path: Path) -> None:
    for step in (1, 2):
        logger = JsonlLogger(tmp_path / "m.jsonl")
        logger.log({"loss": 1.0}, step=step, split="train")
        logger.close()
    assert len((tmp_path / "m.jsonl").read_text("utf-8").splitlines()) == 2


def test_stdout_and_multi_logger() -> None:
    buffer = io.StringIO()
    multi = MultiLogger([StdoutLogger(buffer), StdoutLogger(buffer)])
    multi.log({"loss": 0.123456, "epoch": 2}, step=7, split="valid")
    multi.close()
    out = buffer.getvalue()
    assert out.count("[valid       7]") == 2 and "loss=0.1235" in out and "epoch=2" in out


def test_build_loggers(tmp_path: Path) -> None:
    multi = build_loggers(["jsonl", "stdout"], tmp_path, run_name="r", config={})
    multi.log({"x": 1}, step=1, split="train")
    multi.close()
    assert (tmp_path / "metrics.jsonl").is_file()
    with pytest.raises(ValueError):
        build_loggers(["slack"], tmp_path, run_name="r", config={})
