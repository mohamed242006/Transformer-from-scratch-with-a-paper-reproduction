"""Greedy and beam-search decoding, checkpoint averaging, the translate API and the
evaluation pipeline.

Owner: Evaluation owner. Contract: docs/REPOSITORY.md#srcaiayninference.
"""

from aiayn.inference.averaging import (
    average_checkpoint_files,
    average_state_dicts,
    load_weights,
    snapshot_paths,
)
from aiayn.inference.decoding import (
    DecodeConfig,
    Hypothesis,
    beam_search,
    greedy_decode,
    length_penalty,
)
from aiayn.inference.evaluate import evaluate_run
from aiayn.inference.translator import RunInfo, Translator, load_run

__all__ = [
    "DecodeConfig",
    "Hypothesis",
    "RunInfo",
    "Translator",
    "average_checkpoint_files",
    "average_state_dicts",
    "beam_search",
    "evaluate_run",
    "greedy_decode",
    "length_penalty",
    "load_run",
    "load_weights",
    "snapshot_paths",
]
