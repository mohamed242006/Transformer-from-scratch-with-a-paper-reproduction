"""Decoding under mixed-precision autocast, as evaluation runs on a GPU (ADR-014, ADR-023).

Regression: the incremental state (the Transformer's cached cross-attention keys/values)
was built outside autocast. On the T4 the fp16 decoder queries then met fp32 keys and
every evaluation crashed. CPU tests never used autocast, so this suite (part of the Kaggle
preflight) decodes with fp16 on CUDA when available, else bf16 on CPU.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from aiayn.inference.decoding import DecodeConfig, beam_search, greedy_decode
from tests.unit.test_incremental_decoding import MODELS

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
DTYPE = torch.float16 if CUDA else torch.bfloat16
# oneDNN has no bf16 LSTM kernel on many CPUs; on CUDA every model is covered.
CPU_UNSUPPORTED = {"lstm", "lstm_luong"}


@pytest.mark.numerical
@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("beam", [1, 4])
def test_incremental_decoding_under_autocast(name: str, beam: int) -> None:
    if not CUDA and name in CPU_UNSUPPORTED:
        pytest.skip("bf16 LSTM is not supported by oneDNN on this CPU")
    model: Any = MODELS[name]().to(DEVICE).eval()
    torch.manual_seed(0)
    src = torch.randint(4, 40, (3, 7), device=DEVICE)
    src[:, -1] = 3
    config = DecodeConfig(beam_size=beam, max_len_b=8)
    search = greedy_decode if beam == 1 else beam_search
    hyps = search(model, src, config, autocast_dtype=DTYPE, incremental=True)
    assert len(hyps) == 3
    assert all(all(0 < t < 40 for t in h.tokens) for h in hyps)
    assert all(torch.isfinite(torch.tensor(h.score)) for h in hyps)
