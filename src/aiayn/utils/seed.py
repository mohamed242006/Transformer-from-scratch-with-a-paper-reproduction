"""Global seeding and determinism controls."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (all devices).

    With ``deterministic=True`` PyTorch is forced onto deterministic kernels; operations
    without a deterministic implementation then raise instead of silently varying.
    Note: ``PYTHONHASHSEED`` only takes effect if set before the interpreter starts.
    Data-pipeline randomness does not depend on this function: samplers and synthetic
    datasets use their own explicit seeds (ADR-009).
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
