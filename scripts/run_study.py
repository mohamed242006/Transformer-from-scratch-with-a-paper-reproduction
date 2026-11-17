"""Run a multi-seed study within a session budget (ADR-022).

Kaggle (see notebooks/kaggle_study.ipynb):
    python scripts/run_study.py --study configs/study/week8.yaml \\
        --work /kaggle/working/runs --input /kaggle/input --unzip-into /tmp/aiayn-restored \\
        --minutes 640 --reserve 30

Local dry run (prints the task list, trains nothing):
    python scripts/run_study.py --study configs/study/week8.yaml --work runs --dry-run

Exit codes: 0 = study complete; 3 = budget reached, run again to continue.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from aiayn.study import StudyEnv, collect_states, load_study, run_study, tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path, help="writable runs directory")
    parser.add_argument("--input", type=Path, action="append", default=[], help="read-only roots")
    parser.add_argument("--unzip-into", type=Path, help="where zipped inputs are extracted")
    parser.add_argument("--minutes", type=float, default=640.0, help="session time available")
    parser.add_argument("--reserve", type=float, default=30.0, help="minutes kept for reporting")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    study = load_study(args.study)
    env = StudyEnv(args.work, args.input, args.unzip_into, Path.cwd(), sys.executable)
    if args.dry_run:
        todo = tasks(study, collect_states(study, env))
        print(f"{len(todo)} tasks: {dict(Counter(t.phase for t in todo))}")
        for t in todo[:40]:
            print(f"  [{t.phase}] {t.run.name} train={t.train} evals={[s.tag for s in t.evals]}")
        return 0
    outcome = run_study(study, env, minutes=args.minutes, reserve_minutes=args.reserve)
    print(
        f"study {outcome.status}: trained {len(outcome.trained)}, evaluations "
        f"{len(outcome.evaluated)}, failed {outcome.failed or 'none'}, "
        f"{outcome.remaining_tasks} tasks remaining"
    )
    return 0 if outcome.status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
