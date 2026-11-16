"""Average model weights (paper §6.1): a run's last N snapshots, or explicit files.

python scripts/average_checkpoints.py --run-dir runs/x --last 5 --output runs/x/averaged.pt
python scripts/average_checkpoints.py --inputs a.pt b.pt c.pt --output avg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aiayn.inference import average_checkpoint_files, snapshot_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--last", type=int, help="with --run-dir: newest N snapshots (default all)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = snapshot_paths(args.run_dir, args.last) if args.run_dir else list(args.inputs)
    averaged = average_checkpoint_files(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, args.output)
    print(f"averaged {len(paths)} checkpoints -> {args.output}")
    for path in paths:
        print("  ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
