"""Download, verify and extract a parallel corpus described by a data config.

Usage:
    python scripts/prepare_data.py --config configs/data/multi30k.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aiayn.data.config import SPLITS, load_data_config
from aiayn.data.datasets import read_lines
from aiayn.data.prepare import prepare_translation_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--mirror-dir",
        action="append",
        default=[],
        type=Path,
        help="directory holding the raw archives (offline / Kaggle input); checksums still apply",
    )
    args = parser.parse_args(argv)

    config = load_data_config(args.config, tuple(args.override))
    if config.translation is None:
        print(f"{config.name}: synthetic task, nothing to prepare.")
        return 0

    paths = prepare_translation_data(config, mirror_dirs=args.mirror_dir)
    t = config.translation
    print(f"Prepared {config.name} -> {paths.processed_dir}")
    for split in SPLITS:
        n = len(read_lines(paths.text_file(split, t.src_lang)))
        print(f"  {split:<5} {n:>9,} sentence pairs ({t.src_lang}->{t.tgt_lang})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
