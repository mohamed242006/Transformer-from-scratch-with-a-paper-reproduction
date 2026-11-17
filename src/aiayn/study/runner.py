"""Execute a study within a session budget (ADR-022).

Loop: index runs (working directory first, then read-only inputs) -> next task -> make
the run local (copy from inputs if needed) -> train (``scripts/train.py`` subprocess, so
signals, budgets and exact resume behave exactly as in single runs) -> evaluate
(in-process) -> slim when completed. A run that crashes or diverges is marked
``FAILED.json`` and reported, never retried. When the budget is nearly used up the loop
stops; the next session continues from the published runs.

Every run the study touches ends up in the working directory, so one session's output
(and the ``aiayn-runs`` dataset version published from it) is a complete snapshot.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from aiayn.diagnostics.kaggle_runs import index_runs
from aiayn.inference.evaluate import evaluate_run
from aiayn.study.config import StudyConfig
from aiayn.study.plan import EvalSpec, PlannedRun, Task, all_run_names, main_run, tasks
from aiayn.study.storage import RunState, mark_failed, read_state, slim_run

__all__ = ["StudyEnv", "StudyOutcome", "adopt_inputs", "collect_states", "run_study"]

EXIT_RESUMABLE = 3
MIN_TRAIN_MINUTES = 8.0  # do not start (or resume) training with less time than this
MIN_EVAL_MINUTES = 1.5

TrainFn = Callable[[PlannedRun, Path, bool, int], int]
EvalFn = Callable[[Path, EvalSpec], None]


@dataclass
class StudyEnv:
    work_root: Path  # writable: /kaggle/working/runs or runs/
    input_roots: list[Path] = field(default_factory=list)  # read-only: /kaggle/input
    unzip_into: Path | None = None
    repo_root: Path = Path(".")
    python: str = sys.executable


@dataclass
class StudyOutcome:
    status: str  # "complete" | "budget"
    trained: list[str] = field(default_factory=list)
    evaluated: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    remaining_tasks: int = 0


def collect_states(
    study: StudyConfig, env: StudyEnv, input_index: Mapping[str, Path] | None = None
) -> dict[str, RunState]:
    """State of every study run: the working copy wins over input copies."""
    if input_index is None:
        input_index = index_runs(env.input_roots, unzip_into=env.unzip_into)
    states = {}
    for name in all_run_names(study):
        local = env.work_root / name
        if (local / "metadata.json").is_file() or (local / "FAILED.json").is_file():
            states[name] = read_state(local)
        elif name in input_index:
            states[name] = read_state(input_index[name])
    return states


def default_train(env: StudyEnv) -> TrainFn:
    def train(run: PlannedRun, run_dir: Path, resume: bool, budget_minutes: int) -> int:
        cmd = [env.python, "-u", "scripts/train.py", "--run-dir", str(run_dir), "--prepare-data"]
        if resume:
            cmd += ["--resume", str(run_dir)]
        else:
            cmd += ["--experiment", str(run.experiment)]
            for item in run.overrides:
                cmd += ["--override", item]
        cmd += ["--override", f"train.time_budget_minutes={budget_minutes}"]
        print("$", " ".join(cmd), flush=True)
        return subprocess.run(cmd, cwd=env.repo_root).returncode

    return train


def default_eval(run_dir: Path, spec: EvalSpec) -> None:
    result = evaluate_run(
        run_dir,
        split=spec.split,  # type: ignore[arg-type]
        checkpoint=spec.checkpoint,
        decode=spec.decode,
        prepare_data=True,
    )
    print(f"  {run_dir.name} {spec.tag}: BLEU {result['bleu']:.2f}", flush=True)


def _materialize(task: Task, states: Mapping[str, RunState], env: StudyEnv) -> Path:
    local = env.work_root / task.run.name
    state = states.get(task.run.name)
    if state is not None and state.path.resolve() != local.resolve():
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(state.path, local)  # read-only input -> working copy
    return local


def adopt_inputs(study: StudyConfig, env: StudyEnv, input_index: Mapping[str, Path]) -> list[str]:
    """Copy study runs that exist only in the inputs into the working directory (slimmed
    when complete), so this session's output is a complete snapshot of the study."""
    adopted = []
    main_names = {main_run(study, m, s).name for m in study.models for s in study.main_seeds}
    for name, state in collect_states(study, env, input_index).items():
        local = env.work_root / name
        if state.path.resolve() == local.resolve():
            continue
        shutil.copytree(state.path, local)
        if state.completed and not state.failed:
            slim_run(local, keep_best=name in main_names)
        adopted.append(name)
    return adopted


def run_study(
    study: StudyConfig,
    env: StudyEnv,
    *,
    minutes: float,
    reserve_minutes: float = 25.0,
    train_fn: TrainFn | None = None,
    eval_fn: EvalFn | None = None,
    clock: Callable[[], float] = time.time,
    max_tasks: int | None = None,
) -> StudyOutcome:
    """Work through ``tasks(study, ...)`` until done or ``minutes - reserve`` elapse."""
    train_fn = train_fn or default_train(env)
    eval_fn = eval_fn or default_eval
    env.work_root.mkdir(parents=True, exist_ok=True)
    deadline = clock() + (minutes - reserve_minutes) * 60
    input_index = index_runs(env.input_roots, unzip_into=env.unzip_into)
    try:
        return _loop(study, env, input_index, deadline, train_fn, eval_fn, clock, max_tasks)
    finally:
        adopt_inputs(study, env, input_index)


def _loop(
    study: StudyConfig,
    env: StudyEnv,
    input_index: Mapping[str, Path],
    deadline: float,
    train_fn: TrainFn,
    eval_fn: EvalFn,
    clock: Callable[[], float],
    max_tasks: int | None,
) -> StudyOutcome:
    outcome = StudyOutcome(status="complete")
    previous: tuple[str, bool, tuple[str, ...]] | None = None
    done = 0
    while True:
        states = collect_states(study, env, input_index)
        todo = tasks(study, states)
        outcome.remaining_tasks = len(todo)
        if not todo:
            return outcome
        if max_tasks is not None and done >= max_tasks:
            outcome.status = "budget"
            return outcome
        task = todo[0]
        signature = (task.run.name, task.train, tuple(s.tag for s in task.evals))
        if signature == previous:
            raise RuntimeError(f"no progress on {task.run.name} ({task.phase}): {signature}")
        previous = signature
        left = (deadline - clock()) / 60
        if left < (MIN_TRAIN_MINUTES if task.train else MIN_EVAL_MINUTES):
            outcome.status = "budget"
            return outcome
        run_dir = _materialize(task, states, env)
        print(
            f"[{task.phase}] {task.run.name}: train={task.train} evals={len(task.evals)}",
            flush=True,
        )
        if task.train:
            resume = (run_dir / "checkpoints" / "last.pt").is_file() or (
                run_dir / "checkpoints" / "last.prev.pt"
            ).is_file()
            code = train_fn(task.run, run_dir, resume, int(left))
            if code == EXIT_RESUMABLE:
                outcome.status = "budget"
                return outcome
            if code != 0:
                mark_failed(run_dir, code, "training exited with an error (see the session log)")
                outcome.failed.append(task.run.name)
                done += 1
                continue
            outcome.trained.append(task.run.name)
        for spec in task.evals:
            if (deadline - clock()) / 60 < MIN_EVAL_MINUTES:
                outcome.status = "budget"
                return outcome
            eval_fn(run_dir, spec)
            outcome.evaluated.append(f"{task.run.name}:{spec.tag}")
        if read_state(run_dir).completed:
            slim_run(run_dir, keep_best=task.run.keep_best)
        done += 1
