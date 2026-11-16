"""Validation report, version parsing and Kaggle packaging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiayn.diagnostics.packaging import (
    MANIFEST_NAME,
    build_code_bundle,
    build_data_bundle,
    code_files,
    tree_hash,
)
from aiayn.diagnostics.report import Outcome, Report
from aiayn.diagnostics.validation import _version_tuple
from tests.conftest import REPO_ROOT


def test_report_records_statuses_and_exit_code(tmp_path: Path) -> None:
    report = Report()
    report.run("a", lambda: Outcome("pass", "ok", {"x": 1}))
    report.run("b", lambda: Outcome("warn", "meh"), critical=False)
    report.skip("c", "not applicable")
    assert report.exit_code() == 0 and report.passed("a") and report.passed("b")
    report.run("d", lambda: Outcome("fail", "broken"), critical=False)
    assert report.exit_code() == 0  # non-critical failures do not fail the run
    report.run("e", lambda: Outcome("fail", "broken"))
    assert report.exit_code() == 1
    data = json.loads(report.write_json(tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["ok"] is False and data["counts"] == {"pass": 1, "warn": 1, "fail": 2, "skip": 1}
    assert "FAILED" in report.format_table()


def test_exceptions_become_failures_with_traceback() -> None:
    report = Report()

    def boom() -> Outcome:
        raise RuntimeError("kaboom")

    result = report.run("x", boom)
    assert result.status == "fail" and "kaboom" in result.message
    assert any("RuntimeError" in line for line in result.details["traceback"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2.4.1+cu121", (2, 4, 1)),
        ("0.2.0", (0, 2, 0)),
        ("2.15.0+cpu", (2, 15, 0)),
        ("6.0.1", (6, 0, 1)),
    ],
)
def test_version_parsing(text: str, expected: tuple[int, ...]) -> None:
    assert _version_tuple(text) == expected


def test_code_files_exclude_environments_data_and_caches() -> None:
    files = {p.as_posix() for p in code_files(REPO_ROOT)}
    assert "pyproject.toml" in files and "src/aiayn/models/transformer.py" in files
    assert "data/README.md" not in files  # the top-level data/ tree stays out of the bundle
    # Regression (Week 5 dry run): "data" must only match the top-level folder, never the
    # aiayn.data package or configs/data/.
    assert "src/aiayn/data/tokenizer.py" in files
    assert "configs/data/multi30k.yaml" in files
    forbidden = (".venv/", "__pycache__", ".mypy_cache/", ".pytest_cache/", ".egg-info", "dist/")
    assert not [f for f in files if any(token in f for token in forbidden)]


def test_tree_hash_is_order_independent_and_content_sensitive() -> None:
    a = {"x.py": "1" * 64, "y.py": "2" * 64}
    b = dict(reversed(list(a.items())))
    assert tree_hash(a) == tree_hash(b)
    assert tree_hash(a) != tree_hash({**a, "y.py": "3" * 64})


def test_code_bundle_is_deterministic(tmp_path: Path) -> None:
    first = build_code_bundle(REPO_ROOT, tmp_path / "one", kaggle_user="someone")
    second = build_code_bundle(REPO_ROOT, tmp_path / "two", kaggle_user="someone")
    assert first.tree_sha256 == second.tree_sha256
    manifest = json.loads((tmp_path / "one" / "aiayn" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["tree_sha256"] == first.tree_sha256
    assert (tmp_path / "one" / "aiayn" / "pyproject.toml").is_file()
    meta = json.loads((tmp_path / "one" / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/aiayn-code"


def test_data_bundle_requires_a_translation_config(tmp_path: Path) -> None:
    from aiayn.data.config import load_data_config

    copy_cfg = load_data_config(REPO_ROOT / "configs" / "data" / "copy_task.yaml")
    with pytest.raises(ValueError, match="translation"):
        build_data_bundle(copy_cfg, tmp_path, kaggle_user="someone")


def test_code_bundle_is_importable_and_complete(tmp_path: Path) -> None:
    """Every Python module and config of the working copy must be in the bundle."""
    build_code_bundle(REPO_ROOT, tmp_path, kaggle_user="someone")
    bundle = tmp_path / "aiayn"
    for pattern in ("src/aiayn/**/*.py", "configs/**/*.yaml", "scripts/*.py", "tests/**/*.py"):
        expected = {
            p.relative_to(REPO_ROOT)
            for p in REPO_ROOT.glob(pattern)
            if "__pycache__" not in p.parts
        }
        shipped = {p.relative_to(bundle) for p in bundle.glob(pattern)}
        assert expected == shipped, (pattern, expected ^ shipped)
