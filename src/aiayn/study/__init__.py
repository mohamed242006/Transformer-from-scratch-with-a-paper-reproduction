"""Multi-seed experimental studies: plan, execution within a session budget, slimming of
completed runs and the statistical report (ADR-022).

Owner: Research owner. Contract: docs/REPOSITORY.md#srcaiaynstudy. Sits on top of
training, inference and evaluation; only scripts and notebooks import it.
"""

from aiayn.study.analysis import build_study_report, write_study_report
from aiayn.study.config import StudyConfig, Variant, VariantGroup, load_study
from aiayn.study.plan import EvalSpec, PlannedRun, Task, planned_runs, tasks
from aiayn.study.runner import StudyEnv, StudyOutcome, collect_states, run_study
from aiayn.study.storage import RunState, read_state, slim_run

__all__ = [
    "EvalSpec",
    "PlannedRun",
    "RunState",
    "StudyConfig",
    "StudyEnv",
    "StudyOutcome",
    "Task",
    "Variant",
    "VariantGroup",
    "build_study_report",
    "collect_states",
    "load_study",
    "planned_runs",
    "read_state",
    "run_study",
    "slim_run",
    "tasks",
    "write_study_report",
]
