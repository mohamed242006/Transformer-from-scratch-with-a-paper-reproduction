"""Run discovery, staging and publishing for Kaggle persistence (ADR-020).

The Kaggle CLI is replaced by a fake runner: nothing is uploaded by these tests.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from aiayn.diagnostics.kaggle_runs import (
    find_runs,
    kaggle_credentials_available,
    latest_run,
    publish_dataset,
    run_progress,
    stage_runs,
)


def make_run(root: Path, name: str, end_update: int) -> Path:
    run = root / name
    (run / "checkpoints" / "avg").mkdir(parents=True)
    (run / "config.yaml").write_text("name: x\n", encoding="utf-8")
    meta = {"sessions": [{"start_update": 0, "end_update": end_update}]}
    (run / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    for name_ in ("last.pt", "last.prev.pt", "best.pt", "last.pt.tmp"):
        (run / "checkpoints" / name_).write_bytes(b"x")
    (run / "checkpoints" / "avg" / "step_00000010.pt").write_bytes(b"x")
    return run


def test_progress_and_latest_copy_wins(tmp_path: Path) -> None:
    older = make_run(tmp_path / "notebook-output" / "runs", "exp_s1", 100)
    newer = make_run(tmp_path / "aiayn-runs" / "runs", "exp_s1", 250)
    make_run(tmp_path / "aiayn-runs" / "runs", "other_s1", 999)
    assert run_progress(older) == 100 and run_progress(tmp_path / "missing") == 0
    assert set(find_runs("exp_s1", [tmp_path])) == {older, newer}
    assert latest_run("exp_s1", [tmp_path]) == newer
    assert latest_run("absent", [tmp_path, tmp_path / "nope"]) is None


def test_runs_inside_zip_archives_are_found(tmp_path: Path) -> None:
    run = make_run(tmp_path / "src", "zipped_s1", 42)
    inputs = tmp_path / "input"
    inputs.mkdir()
    with zipfile.ZipFile(inputs / "runs.zip", "w") as zf:
        for path in run.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(run.parent))
    assert latest_run("zipped_s1", [inputs]) is None  # not found without extraction
    found = latest_run("zipped_s1", [inputs], unzip_into=tmp_path / "restored")
    assert found is not None and run_progress(found) == 42


def test_staging_excludes_fallbacks_and_keeps_previous_runs(tmp_path: Path) -> None:
    previous = tmp_path / "attached"
    make_run(previous / "runs", "keep_s1", 10)
    make_run(previous / "runs", "exp_s1", 5)  # stale copy: replaced by the new one
    current = make_run(tmp_path / "working" / "runs", "exp_s1", 50)
    staging = stage_runs([current], tmp_path / "staging", kaggle_user="me", previous=previous)
    staged = staging / "runs"
    assert sorted(p.name for p in staged.iterdir()) == ["exp_s1", "keep_s1"]
    assert run_progress(staged / "exp_s1") == 50
    ckpts = {p.name for p in (staged / "exp_s1" / "checkpoints").iterdir()}
    assert ckpts == {"last.pt", "best.pt", "avg"}  # no last.prev.pt, no partial files
    meta = json.loads((staging / "dataset-metadata.json").read_text("utf-8"))
    assert meta["id"] == "me/aiayn-runs"


def test_credentials_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    assert not kaggle_credentials_available()
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    assert kaggle_credentials_available()
    monkeypatch.delenv("KAGGLE_KEY")
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}", encoding="utf-8")
    assert kaggle_credentials_available()


class FakeKaggle:
    def __init__(self, version_rc: int, version_err: str = "") -> None:
        self.calls: list[list[str]] = []
        self.version_rc, self.version_err = version_rc, version_err

    def __call__(self, args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args[2] == "version":
            return subprocess.CompletedProcess(args, self.version_rc, "v-out", self.version_err)
        return subprocess.CompletedProcess(args, 0, "created", "")


@pytest.fixture
def creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setenv("KAGGLE_KEY", "secret")


def test_publish_versions_an_existing_dataset(tmp_path: Path, creds: None) -> None:
    fake = FakeKaggle(version_rc=0)
    result = publish_dataset(tmp_path, message="m", runner=fake)
    assert result.ok and result.action == "version" and len(fake.calls) == 1
    assert fake.calls[0][:3] == ["kaggle", "datasets", "version"] and "--dir-mode" in fake.calls[0]


def test_publish_creates_the_dataset_on_first_use(tmp_path: Path, creds: None) -> None:
    fake = FakeKaggle(version_rc=1, version_err="404 - Not Found")
    result = publish_dataset(tmp_path, message="m", runner=fake)
    assert result.ok and result.action == "create"
    assert [c[2] for c in fake.calls] == ["version", "create"]


def test_publish_reports_other_failures(tmp_path: Path, creds: None) -> None:
    fake = FakeKaggle(version_rc=1, version_err="403 - Forbidden")
    result = publish_dataset(tmp_path, message="m", runner=fake)
    assert not result.ok and len(fake.calls) == 1


def test_publish_without_credentials_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    fake = FakeKaggle(version_rc=0)
    result = publish_dataset(tmp_path, message="m", runner=fake)
    assert result.action == "skipped" and not fake.calls
