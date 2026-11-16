"""Publish run directories to a private Kaggle Dataset so they survive the session (ADR-020).

    python scripts/kaggle_publish_runs.py --run /kaggle/working/runs/multi30k_tiny_s1 \
        --user <kaggle-username> [--previous /kaggle/input/aiayn-runs] [--dry-run]

Needs Kaggle API credentials (KAGGLE_USERNAME / KAGGLE_KEY, e.g. from notebook secrets).
Exit code 0 on success or dry run, 1 if the upload failed, 2 if credentials are missing.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from aiayn.diagnostics.kaggle_runs import publish_dataset, stage_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--user", required=True, help="Kaggle username (dataset owner)")
    parser.add_argument("--slug", default="aiayn-runs")
    parser.add_argument("--previous", type=Path, help="attached runs dataset to carry over")
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path(tempfile.gettempdir()) / "aiayn-runs-upload",
        help="scratch folder outside /kaggle/working (so it is not saved twice)",
    )
    parser.add_argument("--message", default="update runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    staging = stage_runs(
        args.run, args.staging, kaggle_user=args.user, slug=args.slug, previous=args.previous
    )
    staged = sorted(p.name for p in (staging / "runs").iterdir())
    print(f"staged {len(staged)} run(s) in {staging}: {staged}")
    if args.dry_run:
        return 0
    result = publish_dataset(staging, message=args.message)
    print(f"publish: {result.action} ok={result.ok}\n{result.output}")
    if result.action == "skipped":
        return 2
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
