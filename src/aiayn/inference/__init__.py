"""Greedy and beam-search decoding, checkpoint averaging, the translate API, the
evaluation pipeline, run reports and fair cross-run comparison.

Owner: Evaluation owner. Contract: docs/REPOSITORY.md#srcaiayninference.
"""

from aiayn.inference.averaging import (
    average_checkpoint_files,
    average_state_dicts,
    load_weights,
    snapshot_paths,
)
from aiayn.inference.comparison import build_comparison, summarize_run, write_comparison
from aiayn.inference.decoding import (
    DecodeConfig,
    Hypothesis,
    beam_search,
    greedy_decode,
    length_penalty,
)
from aiayn.inference.evaluate import decode_tag, eval_tag, evaluate_run, load_references
from aiayn.inference.report import build_run_report, write_run_report
from aiayn.inference.translator import RunInfo, Translator, load_run

__all__ = [
    "DecodeConfig",
    "Hypothesis",
    "RunInfo",
    "Translator",
    "average_checkpoint_files",
    "average_state_dicts",
    "beam_search",
    "build_comparison",
    "build_run_report",
    "decode_tag",
    "eval_tag",
    "evaluate_run",
    "greedy_decode",
    "length_penalty",
    "load_references",
    "load_run",
    "load_weights",
    "snapshot_paths",
    "summarize_run",
    "write_comparison",
    "write_run_report",
]
