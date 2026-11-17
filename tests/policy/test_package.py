"""Smoke checks for the package skeleton and repository contract."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SUBPACKAGES = [
    "aiayn.utils",
    "aiayn.modules",
    "aiayn.models",
    "aiayn.models.baselines",
    "aiayn.data",
    "aiayn.training",
    "aiayn.inference",
    "aiayn.evaluation",
    "aiayn.tracking",
    "aiayn.diagnostics",
    "aiayn.study",
]


def test_package_has_version() -> None:
    import aiayn

    assert isinstance(aiayn.__version__, str) and aiayn.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports_and_is_documented(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__ and "Owner:" in module.__doc__, f"{name} lacks an ownership docstring"


def test_required_week1_documents_exist() -> None:
    required = [
        "README.md",
        "docs/REPOSITORY.md",
        "docs/paper_notes.md",
        "docs/math.md",
        "docs/experiment_log.md",
        "docs/decisions/README.md",
    ]
    missing = [p for p in required if not (REPO_ROOT / p).is_file()]
    assert not missing, f"missing documents: {missing}"


def test_every_adr_is_indexed() -> None:
    decisions = REPO_ROOT / "docs" / "decisions"
    index = (decisions / "README.md").read_text(encoding="utf-8")
    adrs = sorted(p.name for p in decisions.glob("ADR-*.md"))
    assert adrs, "no ADRs found"
    unindexed = [name for name in adrs if name not in index]
    assert not unindexed, f"ADRs missing from docs/decisions/README.md: {unindexed}"
