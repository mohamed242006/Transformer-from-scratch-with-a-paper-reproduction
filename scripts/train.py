"""Train a Transformer from an experiment config, or resume a run (ADR-013, ADR-018).

New run:
    python scripts/train.py --experiment configs/experiment/multi30k_tiny.yaml \\
        --run-dir runs/multi30k_tiny_s1 [--override train.seed=2] [--prepare-data]

Resume (same machine, or Kaggle: copy a read-only input run to a writable location):
    python scripts/train.py --resume runs/multi30k_tiny_s1 [--override train.max_epochs=40]
    python scripts/train.py --resume /kaggle/input/<output>/runs/x --run-dir /kaggle/working/runs/x

Exit code 0 when training completed, 3 when it stopped early (time budget or signal) and
can be resumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

from aiayn.training import Trainer, ensure_data, load_experiment
from aiayn.training.experiment import experiment_from_dict
from aiayn.utils.config import load_yaml

EXIT_RESUMABLE = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", type=Path, help="configs/experiment/*.yaml")
    source.add_argument("--resume", type=Path, help="existing run directory")
    parser.add_argument("--run-dir", type=Path, help="default: runs/<timestamp>_<name>_s<seed>")
    parser.add_argument("--override", action="append", default=[], metavar="GROUP.KEY=VALUE")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="download/verify the corpus and train the tokenizer if missing",
    )
    parser.add_argument("--data-mirror", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    if args.resume is not None:
        run_dir = args.run_dir or args.resume
        if run_dir.resolve() != args.resume.resolve():
            if run_dir.exists():
                raise SystemExit(f"{run_dir} already exists; refusing to overwrite")
            shutil.copytree(args.resume, run_dir)  # e.g. /kaggle/input -> /kaggle/working
        if args.prepare_data:  # fresh machine/session: data is reproducible from pinned sources
            raw = load_yaml(run_dir / "config.yaml")
            ensure_data(experiment_from_dict(raw), args.data_mirror, tokenizer=False)
        trainer = Trainer.resume(run_dir, args.override)
        print(
            f"resumed {run_dir} at update {trainer.progress.update}, epoch {trainer.progress.epoch}"
        )
    else:
        experiment = load_experiment(args.experiment, args.override)
        if args.prepare_data:
            ensure_data(experiment, args.data_mirror)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = (
            args.run_dir or Path("runs") / f"{stamp}_{experiment.name}_s{experiment.train.seed}"
        )
        trainer = Trainer.create(experiment, run_dir)
        print(f"new run {run_dir}: device {trainer.device}, amp {trainer.amp_dtype}")

    result = trainer.train()
    print(
        f"status={result.status} updates={result.updates} epochs={result.epochs_completed} "
        f"best_valid_nll={result.best_valid_nll} run_dir={result.run_dir}"
    )
    return 0 if result.status == "completed" else EXIT_RESUMABLE


if __name__ == "__main__":
    raise SystemExit(main())
