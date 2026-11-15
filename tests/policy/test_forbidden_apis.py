"""Enforce ADR-001: no high-level Transformer APIs inside ``src/``.

The check is AST-based, so names that appear only in docstrings or comments are
not flagged. It detects:

* attribute access such as ``nn.MultiheadAttention`` or ``torch.nn.functional.layer_norm``
  when the attribute is reached through ``nn``, ``F``, ``functional`` or ``torch.nn``;
* ``from torch.nn import LayerNorm`` style imports;
* any import of Hugging Face ``transformers``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "Transformer",
        "TransformerEncoder",
        "TransformerDecoder",
        "TransformerEncoderLayer",
        "TransformerDecoderLayer",
        "MultiheadAttention",
        "scaled_dot_product_attention",
        "multi_head_attention_forward",
        "LayerNorm",
        "layer_norm",
    }
)

# Attribute chains whose prefix resolves to torch.nn / torch.nn.functional.
TORCH_NN_PREFIXES: frozenset[str] = frozenset(
    {"nn", "F", "functional", "torch.nn", "torch.nn.functional", "nn.functional"}
)

FORBIDDEN_MODULE_ROOTS: frozenset[str] = frozenset({"transformers"})


def _dotted_name(node: ast.AST) -> str | None:
    """Return ``a.b.c`` for nested Attribute/Name nodes, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def find_violations(source: str, filename: str = "<string>") -> list[str]:
    """Return human-readable violations of ADR-001 found in ``source``."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            dotted = _dotted_name(node)
            if dotted is not None:
                prefix = dotted.rsplit(".", 1)[0]
                if prefix in TORCH_NN_PREFIXES:
                    violations.append(f"{filename}:{node.lineno}: uses `{dotted}`")

        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULE_ROOTS:
                violations.append(f"{filename}:{node.lineno}: imports from `{node.module}`")
            elif node.module.startswith("torch.nn"):
                for alias in node.names:
                    if alias.name in FORBIDDEN_NAMES:
                        violations.append(
                            f"{filename}:{node.lineno}: imports `{alias.name}` from `{node.module}`"
                        )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULE_ROOTS:
                    violations.append(f"{filename}:{node.lineno}: imports `{alias.name}`")

    return violations


# --------------------------------------------------------------------------- #
# Self-tests: prove the checker catches what it should and nothing more.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "snippet",
    [
        "import torch.nn as nn\nx = nn.MultiheadAttention(8, 2)",
        "import torch\nx = torch.nn.LayerNorm(8)",
        "import torch.nn.functional as F\ny = F.scaled_dot_product_attention(q, k, v)",
        "import torch\ny = torch.nn.functional.layer_norm(x, (8,))",
        "from torch.nn import TransformerEncoderLayer",
        "from torch.nn.functional import layer_norm",
        "import transformers",
        "from transformers import AutoModel",
    ],
)
def test_checker_flags_forbidden_usage(snippet: str) -> None:
    assert find_violations(snippet), f"checker missed a violation in:\n{snippet}"


@pytest.mark.parametrize(
    "snippet",
    [
        '"""We do not use nn.MultiheadAttention here (see ADR-001)."""',
        "# nn.LayerNorm is forbidden",
        "import torch.nn as nn\nlin = nn.Linear(8, 8)\ndrop = nn.Dropout(0.1)",
        "class LayerNorm:\n    pass\nln = LayerNorm()",
        "from aiayn.models import Transformer\nm = Transformer()",
        "import aiayn.modules as modules\nx = modules.LayerNorm",
    ],
)
def test_checker_allows_legitimate_code(snippet: str) -> None:
    assert find_violations(snippet) == []


# --------------------------------------------------------------------------- #
# The actual policy check over the source tree.
# --------------------------------------------------------------------------- #
def test_src_contains_no_forbidden_apis() -> None:
    files = sorted(SRC_ROOT.rglob("*.py"))
    assert files, f"no Python files found under {SRC_ROOT}"
    violations: list[str] = []
    for path in files:
        violations += find_violations(
            path.read_text(encoding="utf-8"), str(path.relative_to(SRC_ROOT))
        )
    assert not violations, "ADR-001 violations:\n" + "\n".join(violations)
