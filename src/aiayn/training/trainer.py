"""The training loop (ADR-013 resumption, ADR-014 precision, ADR-017 update semantics).

Typical use::

    trainer = Trainer.create(load_experiment("configs/experiment/multi30k_tiny.yaml"), run_dir)
    result = trainer.train()                      # may stop early on the session budget
    trainer = Trainer.resume(run_dir)             # next Kaggle session
    result = trainer.train()

Guarantees (tested on CPU): a run interrupted at any update boundary and resumed produces
bit-identical batches, dropout masks, losses and parameters to the uninterrupted run.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import os
import platform
import shutil
import signal
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch

from aiayn import __version__
from aiayn.data.collate import Batch
from aiayn.data.loading import build_dataloader, build_dataset, train_tokenizer
from aiayn.data.prepare import data_paths, prepare_translation_data
from aiayn.data.sampler import TokenBudgetBatchSampler
from aiayn.data.special_tokens import PAD_ID
from aiayn.data.tokenizer import SentencePieceTokenizer
from aiayn.models import EncoderDecoder, build_model
from aiayn.tracking.loggers import MultiLogger, build_loggers
from aiayn.training.checkpoint import (
    atomic_save,
    capture_rng_state,
    load_training_state,
    prune_snapshots,
    restore_rng_state,
    save_training_state,
)
from aiayn.training.experiment import Experiment, check_compatibility, experiment_from_dict
from aiayn.training.losses import label_smoothed_nll
from aiayn.training.schedule import LRSchedule
from aiayn.utils.config import ConfigError, load_yaml
from aiayn.utils.hashing import directory_tree_hash
from aiayn.utils.params import count_parameters
from aiayn.utils.seed import seed_everything

__all__ = ["Progress", "TrainResult", "Trainer", "ensure_data", "resolve_amp", "resolve_device"]

Status = Literal["completed", "budget_exhausted", "interrupted"]
CHECKPOINT_FORMAT = 1


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def resolve_device(preference: str) -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise ConfigError("train.device=cuda but CUDA is not available")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def resolve_amp(mode: str, device: torch.device) -> torch.dtype | None:
    """ADR-014: fp16 + GradScaler on CUDA >= 7.0 (T4); fp32 otherwise unless requested."""
    if mode == "fp32":
        return None
    if mode == "fp16":
        if device.type != "cuda":
            raise ConfigError("train.amp=fp16 requires a CUDA device")
        return torch.float16
    if mode == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ConfigError("train.amp=bf16 is not supported by this GPU (T4/P100: use fp16)")
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
        return torch.float16
    return None


def ensure_data(
    experiment: Experiment, mirror_dirs: Sequence[Path] = (), *, tokenizer: bool = True
) -> None:
    """Prepare (download/verify) the corpus and, if ``tokenizer``, train the tokenizer when
    missing. Resumed runs pass ``tokenizer=False``: they use the run's own tokenizer copy."""
    cfg = experiment.data
    if cfg.translation is None:
        return
    paths = data_paths(cfg)
    if not paths.text_file("train", cfg.translation.src_lang).is_file():
        prepare_translation_data(cfg, mirror_dirs=mirror_dirs)
    if tokenizer and not paths.tokenizer_model.is_file():
        train_tokenizer(cfg)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Progress:
    """Everything that defines *where* training is; saved in every checkpoint."""

    update: int = 0  # successful optimizer updates (= schedule step)
    epoch: int = 0  # current epoch index
    batches_in_epoch: int = 0  # batches of the current epoch already consumed
    tokens: int = 0  # target tokens trained on
    skipped_updates: int = 0  # fp16 overflow skips (ADR-017 §4)
    best_valid_nll: float | None = None
    best_update: int | None = None
    train_seconds: float = 0.0  # accumulated across sessions


@dataclass(frozen=True)
class TrainResult:
    status: Status
    run_dir: Path
    updates: int
    epochs_completed: int
    best_valid_nll: float | None
    last_valid: dict[str, float] | None = None


@dataclass
class _Interval:
    loss: float = 0.0
    nll: float = 0.0
    correct: int = 0
    tokens: int = 0
    updates: int = 0
    started: float = field(default_factory=time.perf_counter)
    paused: float = 0.0  # validation/checkpoint time, excluded from throughput


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Owns model, optimizer, schedule, scaler, data and the run directory."""

    def __init__(self, experiment: Experiment, run_dir: Path, *, resumed: bool) -> None:
        cfg = experiment.train
        check_compatibility(experiment)
        self.experiment = experiment
        self.run_dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.device = resolve_device(cfg.device)
        self.amp_dtype = resolve_amp(cfg.amp, self.device)
        seed_everything(cfg.seed, deterministic=cfg.deterministic)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Data and tokenizer. The run keeps its own tokenizer copy (ADR-013).
        self.tokenizer: SentencePieceTokenizer | None = None
        if experiment.data.translation is not None:
            self.tokenizer = self._run_tokenizer(resumed)
            vocab_size = self.tokenizer.vocab_size
        else:
            assert experiment.data.copy is not None
            vocab_size = experiment.data.copy.vocab_size
        self.train_set = build_dataset(experiment.data, "train", tokenizer=self.tokenizer)
        self.valid_set = build_dataset(experiment.data, "valid", tokenizer=self.tokenizer)
        pin = self.device.type == "cuda"
        # Dedicated generators: DataLoader must not consume the global RNG (exact resume).
        self.train_loader = build_dataloader(
            experiment.data,
            self.train_set,
            "train",
            pin_memory=pin,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        self.valid_loader = build_dataloader(
            experiment.data,
            self.valid_set,
            "valid",
            pin_memory=pin,
            generator=torch.Generator().manual_seed(cfg.seed + 1),
        )
        self.sampler = cast(TokenBudgetBatchSampler, self.train_loader.batch_sampler)

        # Model and optimization.
        self.model: EncoderDecoder = build_model(
            experiment.model, vocab_size=vocab_size, pad_id=PAD_ID, seed=cfg.seed
        ).to(self.device)
        o = cfg.optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0,  # set before every update by the schedule
            betas=(o.betas[0], o.betas[1]),
            eps=o.eps,
            weight_decay=o.weight_decay,
        )
        self.schedule = LRSchedule(cfg.schedule, experiment.model.d_model)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        self.progress = Progress()
        self.last_valid: dict[str, float] | None = None
        self._stop_signal = False
        self._interval = _Interval()
        self._last_grad_norm = float("nan")
        self._last_lr = 0.0
        self._session_start = time.perf_counter()
        self._time_mark = time.perf_counter()
        self.logger: MultiLogger = build_loggers(
            cfg.loggers, run_dir, run_name=run_dir.name, config=experiment.to_dict()
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, experiment: Experiment, run_dir: Path) -> Trainer:
        """Start a new run in an empty (or non-existent) run directory."""
        if (run_dir / "checkpoints").exists():
            raise FileExistsError(f"{run_dir} already has checkpoints; use Trainer.resume")
        trainer = cls(experiment, run_dir, resumed=False)
        (run_dir / "config.yaml").write_text(_yaml_dump(experiment.to_dict()), encoding="utf-8")
        trainer._write_metadata(new=True)
        return trainer

    @classmethod
    def resume(cls, run_dir: Path, overrides: Sequence[str] = ()) -> Trainer:
        """Continue a run from ``last.pt`` (fallback ``last.prev.pt``). Only resume-mutable
        keys (ADR-018) may be overridden; anything else changes the hash and is refused."""
        experiment = experiment_from_dict(load_yaml(run_dir / "config.yaml"), overrides)
        trainer = cls(experiment, run_dir, resumed=True)
        state, path = load_training_state(trainer.ckpt_dir, map_location="cpu")
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ConfigError(f"{path}: unsupported checkpoint format {state.get('format')}")
        if state["config_hash"] != experiment.config_hash:
            raise ConfigError(
                f"{path}: the experiment changed since this checkpoint (config hash mismatch); "
                "only train.{max_updates, max_epochs, time_budget_minutes, log_every, loggers, "
                "device} may change on resume"
            )
        fingerprint = trainer.tokenizer.fingerprint if trainer.tokenizer else None
        if state.get("tokenizer_fingerprint") != fingerprint:
            raise ConfigError(f"{path}: tokenizer fingerprint differs from the checkpoint")
        trainer.model.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        if trainer.scaler.is_enabled() and state.get("scaler"):
            trainer.scaler.load_state_dict(state["scaler"])
        trainer.sampler.load_state_dict(state["sampler"])
        trainer.progress = Progress(**state["progress"])
        restore_rng_state(state["rng"])  # last: nothing after this may consume randomness
        trainer._write_metadata(new=False, resumed_from=path.name)
        return trainer

    def _run_tokenizer(self, resumed: bool) -> SentencePieceTokenizer:
        source = data_paths(self.experiment.data).tokenizer_model
        target = self.run_dir / "tokenizer" / source.name
        if not resumed:
            if not source.is_file():
                raise FileNotFoundError(f"{source} missing: run ensure_data() / prepare first")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        tokenizer = SentencePieceTokenizer(target)
        assert self.experiment.data.translation is not None
        tok_cfg = self.experiment.data.translation.tokenizer
        if tok_cfg.hard_vocab_limit and tokenizer.vocab_size != tok_cfg.vocab_size:
            raise ConfigError(f"tokenizer vocab {tokenizer.vocab_size} != {tok_cfg.vocab_size}")
        return tokenizer

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self) -> TrainResult:
        cfg = self.experiment.train
        status: Status = "completed"
        self._session_start = self._time_mark = time.perf_counter()
        self._interval = _Interval()
        if self._finished():  # e.g. resumed after completion without extending max_updates
            self._write_metadata(new=False, status="completed")
            self.logger.close()
            return self._result("completed")
        with self._signal_handlers():
            while not self._finished():
                stop = self._run_epoch()
                if stop is not None:
                    status = stop
                    break
                self.progress.epoch += 1
                self.progress.batches_in_epoch = 0
                if cfg.validate_every == 0:
                    self.validate()
                self.save_checkpoint()
        if status == "completed":
            if self.last_valid is None or self.last_valid.get("update") != self.progress.update:
                self.validate()
            self.save_checkpoint(snapshot=True)
        else:
            self.save_checkpoint()
        self._write_metadata(new=False, status=status)
        self.logger.close()
        return self._result(status)

    def _finished(self) -> bool:
        cfg = self.experiment.train
        by_updates = cfg.max_updates > 0 and self.progress.update >= cfg.max_updates
        by_epochs = cfg.max_epochs > 0 and self.progress.epoch >= cfg.max_epochs
        return by_updates or by_epochs

    def _run_epoch(self) -> Status | None:
        """One (possibly partial, when resumed) epoch. Returns a stop reason or None."""
        cfg = self.experiment.train
        self.sampler.set_epoch(self.progress.epoch, start_batch=self.progress.batches_in_epoch)
        group: list[Batch] = []
        for batch in self._iterate(self.train_loader):
            group.append(batch)
            if len(group) == cfg.update_freq:
                stop = self._update(group)
                group = []
                if stop is not None:
                    return stop
        if group:  # end of epoch: a smaller final update, as in fairseq
            return self._update(group)
        return None

    @staticmethod
    def _iterate(loader: Any) -> Iterator[Batch]:
        yield from loader

    def _update(self, group: list[Batch]) -> Status | None:
        """One optimizer update over ``group`` (gradient accumulation, ADR-017)."""
        cfg = self.experiment.train
        model, scaler = self.model, self.scaler
        model.train()
        update_tokens = sum(b.num_tgt_tokens for b in group)
        loss_total = torch.zeros((), device=self.device)
        nll_total = torch.zeros((), device=self.device)
        correct_total = torch.zeros((), dtype=torch.long, device=self.device)
        for batch in group:
            b = batch.to(self.device, non_blocking=True)
            with self._autocast():
                logits = model(b.src, b.tgt_in)
            out = label_smoothed_nll(
                logits, b.tgt_out, epsilon=cfg.label_smoothing, ignore_index=PAD_ID
            )
            scaler.scale(out.loss_sum / update_tokens).backward()  # type: ignore[no-untyped-call]
            loss_total += out.loss_sum.detach()
            nll_total += out.nll_sum
            correct_total += out.correct

        scaler.unscale_(self.optimizer)
        max_norm = cfg.clip_norm if cfg.clip_norm > 0 else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if not scaler.is_enabled() and not bool(torch.isfinite(grad_norm)):
            self._dump_nonfinite(group, "gradient norm")
        lr = self.schedule.apply(self.optimizer, self.progress.update + 1)
        scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        skipped = scaler.is_enabled() and scaler.get_scale() < scale_before

        self.progress.batches_in_epoch += len(group)
        if skipped:
            self.progress.skipped_updates += 1
            return self._stop_reason()
        self.progress.update += 1
        self.progress.tokens += update_tokens
        self._last_grad_norm = float(grad_norm)
        self._last_lr = lr
        iv = self._interval
        iv.loss += float(loss_total)
        iv.nll += float(nll_total)
        iv.correct += int(correct_total)
        iv.tokens += update_tokens
        iv.updates += 1
        if not math.isfinite(iv.loss):
            self._dump_nonfinite(group, "loss")
        return self._after_update()

    def _after_update(self) -> Status | None:
        cfg, update = self.experiment.train, self.progress.update
        if update % cfg.log_every == 0:
            self._log_train()
        if cfg.validate_every > 0 and update % cfg.validate_every == 0:
            self.validate()
        if cfg.save_every > 0 and update % cfg.save_every == 0:
            self.save_checkpoint(snapshot=True)
        return self._stop_reason()

    def _stop_reason(self) -> Status | None:
        cfg = self.experiment.train
        if cfg.max_updates > 0 and self.progress.update >= cfg.max_updates:
            return "completed"
        if self._stop_signal:
            return "interrupted"
        budget = cfg.time_budget_minutes * 60.0
        if budget > 0 and time.perf_counter() - self._session_start >= budget:
            return "budget_exhausted"
        return None

    # ------------------------------------------------------------------ #
    # Validation and logging
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        cfg = self.experiment.train
        started = time.perf_counter()
        self.model.eval()
        loss = nll = 0.0
        correct = tokens = 0
        for batch in self._iterate(self.valid_loader):
            b = batch.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(b.src, b.tgt_in)
            out = label_smoothed_nll(
                logits, b.tgt_out, epsilon=cfg.label_smoothing, ignore_index=PAD_ID
            )
            loss += float(out.loss_sum)
            nll += float(out.nll_sum)
            correct += int(out.correct)
            tokens += b.num_tgt_tokens
        self.model.train()
        mean_nll = nll / tokens
        metrics = {
            "update": float(self.progress.update),
            "loss": loss / tokens,
            "nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 50.0)),
            "accuracy": correct / tokens,
        }
        best = self.progress.best_valid_nll
        if best is None or mean_nll < best:
            self.progress.best_valid_nll = mean_nll
            self.progress.best_update = self.progress.update
            atomic_save(
                {
                    "model": self.model.state_dict(),
                    "update": self.progress.update,
                    "valid": metrics,
                },
                self.ckpt_dir / "best.pt",
            )
        metrics["best_nll"] = float(self.progress.best_valid_nll or mean_nll)
        self.last_valid = metrics
        self.logger.log(
            {k: v for k, v in metrics.items() if k != "update"},
            step=self.progress.update,
            split="valid",
        )
        self._interval.paused += time.perf_counter() - started
        return metrics

    def _log_train(self) -> None:
        iv, now = self._interval, time.perf_counter()
        elapsed = max(now - iv.started - iv.paused, 1e-9)
        nll = iv.nll / iv.tokens
        metrics: dict[str, float | int] = {
            "loss": iv.loss / iv.tokens,
            "nll": nll,
            "ppl": math.exp(min(nll, 50.0)),
            "accuracy": iv.correct / iv.tokens,
            "lr": self._last_lr,
            "grad_norm": self._last_grad_norm,
            "tgt_tokens_per_s": round(iv.tokens / elapsed),
            "epoch": self.progress.epoch,
            "skipped": self.progress.skipped_updates,
        }
        if self.scaler.is_enabled():
            metrics["loss_scale"] = self.scaler.get_scale()
        if self.device.type == "cuda":
            metrics["max_mem_mib"] = round(torch.cuda.max_memory_allocated(self.device) / 2**20)
        self.logger.log(metrics, step=self.progress.update, split="train")
        self._interval = _Interval()

    # ------------------------------------------------------------------ #
    # Checkpointing (ADR-013)
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, *, snapshot: bool = False) -> Path:
        now = time.perf_counter()
        self.progress.train_seconds += now - self._time_mark
        self._time_mark = now
        sampler_state = self.sampler.state_dict()
        sampler_state.update(epoch=self.progress.epoch, start_batch=self.progress.batches_in_epoch)
        state = {
            "format": CHECKPOINT_FORMAT,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else {},
            "sampler": sampler_state,
            "progress": asdict(self.progress),
            "rng": capture_rng_state(),
            "config_hash": self.experiment.config_hash,
            "tokenizer_fingerprint": self.tokenizer.fingerprint if self.tokenizer else None,
            "versions": {
                "aiayn": __version__,
                "torch": str(torch.__version__),
            },  # plain str (weights_only)
        }
        path = save_training_state(self.ckpt_dir, state)
        if snapshot:
            avg_dir = self.ckpt_dir / "avg"
            atomic_save(self.model.state_dict(), avg_dir / f"step_{self.progress.update:08d}.pt")
            prune_snapshots(avg_dir, self.experiment.train.average_last)
        self._interval.paused += time.perf_counter() - now
        return path

    def _dump_nonfinite(self, group: list[Batch], what: str) -> None:
        dump = self.run_dir / f"nonfinite_update{self.progress.update + 1}.pt"
        torch.save([{"src": b.src, "tgt_in": b.tgt_in, "tgt_out": b.tgt_out} for b in group], dump)
        raise FloatingPointError(
            f"non-finite {what} at update {self.progress.update + 1}; batch saved to {dump}"
        )

    # ------------------------------------------------------------------ #
    # Run metadata
    # ------------------------------------------------------------------ #
    def _write_metadata(
        self, *, new: bool, status: str | None = None, resumed_from: str | None = None
    ) -> None:
        path = self.run_dir / "metadata.json"
        meta: dict[str, Any] = (
            {} if new or not path.exists() else json.loads(path.read_text("utf-8"))
        )
        if new:
            package_root = Path(__file__).resolve().parents[1]
            meta = {
                "experiment": self.experiment.name,
                "created": _now(),
                "config_hash": self.experiment.config_hash,
                "code_tree_sha256": directory_tree_hash(package_root),
                "tokenizer_fingerprint": self.tokenizer.fingerprint if self.tokenizer else None,
                "parameters": count_parameters(self.model),
                "sessions": [],
            }
        sessions: list[dict[str, Any]] = meta.setdefault("sessions", [])
        if status is None:  # session start
            sessions.append(
                {
                    "start": _now(),
                    "end": None,
                    "status": "running",
                    "start_update": self.progress.update,
                    "resumed_from": resumed_from,
                    "device": self._device_name(),
                    "amp": str(self.amp_dtype),
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                }
            )
        elif sessions:
            sessions[-1].update(
                {"end": _now(), "status": status, "end_update": self.progress.update}
            )
            meta["best_valid_nll"] = self.progress.best_valid_nll
            meta["best_update"] = self.progress.best_update
        _write_json(path, meta)

    def _device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return platform.processor() or "cpu"

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    @contextlib.contextmanager
    def _signal_handlers(self) -> Iterator[None]:
        """SIGTERM/SIGINT request a clean stop at the next update boundary (ADR-013 §5)."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous: dict[int, Any] = {}

        def request_stop(signum: int, frame: Any) -> None:
            self._stop_signal = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                previous[sig] = signal.signal(sig, request_stop)
        try:
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def _result(self, status: Status) -> TrainResult:
        return TrainResult(
            status=status,
            run_dir=self.run_dir,
            updates=self.progress.update,
            epochs_completed=self.progress.epoch,
            best_valid_nll=self.progress.best_valid_nll,
            last_valid=self.last_valid,
        )


def _yaml_dump(data: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False)
