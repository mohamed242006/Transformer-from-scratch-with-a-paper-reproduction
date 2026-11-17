"""Write the study report (STUDY.md + STUDY.json) from run directories (ADR-022).

Works on partial studies (missing runs are reported). Run directories are looked up in
``--work`` first, then in every ``--input`` root.

Usage:
    python scripts/analyze_study.py --study configs/study/week8.yaml --work runs \\
        --out runs/STUDY.md [--input /kaggle/input] [--prepare-data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aiayn.inference import load_references
from aiayn.study import StudyEnv, collect_states, load_study
from aiayn.study.analysis import write_study_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--unzip-into", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-data", action="store_true", help="download data if missing")
    args = parser.parse_args(argv)

    study = load_study(args.study)
    states = collect_states(study, StudyEnv(args.work, args.input, args.unzip_into))
    usable = [s for s in states.values() if s.usable]
    if not usable:
        print("no completed runs yet", file=sys.stderr)
        return 1
    references = load_references(usable[0].path, "test", prepare_data=args.prepare_data)
    out = write_study_report(study, states, references, args.out)
    print(f"wrote {out} and {out.with_suffix('.json')} ({len(usable)} completed runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
