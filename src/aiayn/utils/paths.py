"""Repository-relative path resolution (ADR-007 §4: never depend on the working directory)."""

from __future__ import annotations

from pathlib import Path

__all__ = ["find_repo_root", "resolve_path"]

_MARKER = "pyproject.toml"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the nearest ancestor of ``start`` (default: this file) containing pyproject.toml."""
    here = Path(start) if start is not None else Path(__file__)
    here = here.resolve()
    for candidate in (here, *here.parents):
        if (candidate / _MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"no {_MARKER} found above {here}")


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Absolute paths are returned unchanged; relative ones resolve against ``base``.

    ``base`` defaults to the repository root.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(base) if base is not None else find_repo_root()
    return (root / p).resolve()
