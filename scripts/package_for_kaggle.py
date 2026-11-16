"""Build the Kaggle Dataset bundles (code + data) under dist/kaggle/ (ADR-016).

Usage:
    python scripts/package_for_kaggle.py --kaggle-user YOUR_KAGGLE_USERNAME
    # then, with the Kaggle CLI (or upload the two folders in the web UI):
    kaggle datasets create -p dist/kaggle/aiayn-code --dir-mode zip
    kaggle datasets create -p dist/kaggle/aiayn-multi30k
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aiayn.data.config import load_data_config
from aiayn.diagnostics.packaging import build_code_bundle, build_data_bundle
from aiayn.utils.paths import find_repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-user", default="YOUR_KAGGLE_USERNAME")
    parser.add_argument(
        "--data-config",
        type=Path,
        action="append",
        default=None,
        help="translation configs to bundle (default: multi30k)",
    )
    parser.add_argument("--out", type=Path, default=None, help="default: <repo>/dist/kaggle")
    args = parser.parse_args(argv)

    root = find_repo_root()
    out = args.out or root / "dist" / "kaggle"
    code = build_code_bundle(root, out / "aiayn-code", kaggle_user=args.kaggle_user)
    print(f"code  -> {code.directory}  ({len(code.files)} files, tree {code.tree_sha256[:16]})")
    for path in args.data_config or [root / "configs" / "data" / "multi30k.yaml"]:
        config = load_data_config(path)
        slug = f"aiayn-{config.name.replace('_', '-')}"
        data = build_data_bundle(config, out / slug, kaggle_user=args.kaggle_user)
        print(f"data  -> {data.directory}  ({len(data.files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
