"""Metric loggers behind one small interface (spec §11).

``jsonl`` is the source of truth: one JSON object per line, flushed immediately, so a
killed Kaggle session loses at most the line being written (ADR-013). TensorBoard and
W&B are optional mirrors, imported lazily so the project runs fully offline.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO

__all__ = [
    "JsonlLogger",
    "Logger",
    "MultiLogger",
    "StdoutLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "build_loggers",
]

Scalar = float | int | str


class Logger(Protocol):
    def log(self, metrics: Mapping[str, Scalar], *, step: int, split: str) -> None: ...

    def close(self) -> None: ...


class StdoutLogger:
    """Compact one-line summaries for notebooks and terminals."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout

    def log(self, metrics: Mapping[str, Scalar], *, step: int, split: str) -> None:
        parts = [f"{k}={_fmt(v)}" for k, v in metrics.items()]
        print(f"[{split} {step:>7}] " + "  ".join(parts), file=self.stream, flush=True)

    def close(self) -> None:
        self.stream.flush()


class JsonlLogger:
    """Append-only ``metrics.jsonl`` (crash-safe line by line)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def log(self, metrics: Mapping[str, Scalar], *, step: int, split: str) -> None:
        record: dict[str, Any] = {"time": round(time.time(), 3), "split": split, "step": step}
        record.update({k: _jsonable(v) for k, v in metrics.items()})
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class TensorBoardLogger:
    def __init__(self, log_dir: Path) -> None:
        try:
            module: Any = importlib.import_module("torch.utils.tensorboard")
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("the 'tensorboard' logger needs `pip install tensorboard`") from exc
        self._writer = module.SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: Mapping[str, Scalar], *, step: int, split: str) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._writer.add_scalar(f"{split}/{key}", value, step)

    def close(self) -> None:
        self._writer.close()


class WandbLogger:  # pragma: no cover - network service, exercised manually
    def __init__(self, run_name: str, config: Mapping[str, Any], run_dir: Path) -> None:
        try:
            self._wandb: Any = importlib.import_module("wandb")
        except ImportError as exc:
            raise ImportError("the 'wandb' logger needs `pip install wandb`") from exc
        self._run = self._wandb.init(
            project="aiayn", name=run_name, config=dict(config), dir=str(run_dir), resume="allow"
        )

    def log(self, metrics: Mapping[str, Scalar], *, step: int, split: str) -> None:
        self._run.log({f"{split}/{k}": v for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        self._run.finish()


class MultiLogger:
    def __init__(self, loggers: Sequence[Logger]) -> None:
        self.loggers = list(loggers)

    def log(self, metrics: Mapping[str, Scalar], *, step: int, split: str) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step, split=split)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def build_loggers(
    names: Sequence[str], run_dir: Path, *, run_name: str, config: Mapping[str, Any]
) -> MultiLogger:
    loggers: list[Logger] = []
    for name in names:
        if name == "stdout":
            loggers.append(StdoutLogger())
        elif name == "jsonl":
            loggers.append(JsonlLogger(run_dir / "metrics.jsonl"))
        elif name == "tensorboard":
            loggers.append(TensorBoardLogger(run_dir / "tb"))
        elif name == "wandb":
            loggers.append(WandbLogger(run_name, config, run_dir))
        else:
            raise ValueError(f"unknown logger {name!r}")
    return MultiLogger(loggers)


def _fmt(value: Scalar) -> str:
    if isinstance(value, float):
        return f"{value:.4g}" if math.isfinite(value) else str(value)
    return str(value)


def _jsonable(value: Scalar) -> Scalar | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
