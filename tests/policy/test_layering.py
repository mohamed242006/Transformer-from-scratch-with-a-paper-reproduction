"""Enforce the package import layering (docs/REPOSITORY.md §3) and ADR-005's import rule.

Relative imports are banned by ruff (TID252), so every intra-package dependency appears
as an absolute ``aiayn.<package>`` import that this AST scan can see.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "aiayn"

ALL = {
    "utils",
    "modules",
    "models",
    "data",
    "evaluation",
    "tracking",
    "inference",
    "training",
    "diagnostics",
    "study",
}

ALLOWED: dict[str, set[str]] = {
    "utils": set(),
    "modules": {"utils"},
    "models": {"modules", "utils"},
    "data": {"utils"},
    "evaluation": {"utils"},
    "tracking": {"utils"},
    "inference": {"models", "modules", "data", "evaluation", "utils"},
    "training": ALL - {"training", "diagnostics", "study"},
    # Platform validation/packaging sits on top of everything except studies.
    "diagnostics": ALL - {"diagnostics", "study"},
    # Studies orchestrate everything (run discovery comes from diagnostics); only
    # scripts and notebooks import them.
    "study": ALL - {"study"},
}

# Third-party modules restricted to specific packages.
RESTRICTED_THIRD_PARTY = {
    "sentencepiece": {"data"},
    "sacrebleu": {"evaluation"},
    "sacremoses": {"data", "evaluation"},
}


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def test_allowed_table_covers_every_package() -> None:
    packages = {p.name for p in PACKAGE_ROOT.iterdir() if (p / "__init__.py").is_file()}
    assert packages == set(ALLOWED), "update ALLOWED and docs/REPOSITORY.md together"


def test_import_layering() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT)
        if len(rel.parts) == 1:
            continue  # aiayn/__init__.py
        owner = rel.parts[0]
        for lineno, module in _imports(path):
            parts = module.split(".")
            if parts[0] == "aiayn" and len(parts) > 1:
                target = parts[1]
                if target != owner and target not in ALLOWED[owner]:
                    violations.append(f"{rel}:{lineno}: {owner} -> {target}")
            allowed_owners = RESTRICTED_THIRD_PARTY.get(parts[0])
            if allowed_owners is not None and owner not in allowed_owners:
                violations.append(
                    f"{rel}:{lineno}: {parts[0]} may only be imported in {allowed_owners}"
                )
    assert not violations, "layering violations:\n" + "\n".join(violations)
