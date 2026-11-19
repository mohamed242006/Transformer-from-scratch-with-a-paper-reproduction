"""Write the Week 9 pilot report (PILOT.md, PILOT.json, pilot_curves.csv; ADR-023 §8).

Usage:
    python scripts/pilot_report.py --study configs/study/week9_pilot.yaml \\
        --work runs [--input /kaggle/input] [--validation-report report.json] \\
        --out runs/PILOT.md

Exit codes: 0 = written and Post-LN GO; 2 = written and Post-LN NO-GO;
3 = written, gates still pending; 1 = no pilot runs found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aiayn.study import StudyEnv, collect_states, load_study
from aiayn.study.pilot import write_pilot_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--unzip-into", type=Path)
    parser.add_argument(
        "--validation-report",
        type=Path,
        action="append",
        default=[],
        help="kaggle_validate JSON report (repeatable; later reports override per model)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    study = load_study(args.study)
    states = collect_states(study, StudyEnv(args.work, args.input, args.unzip_into))
    if not states:
        print("no pilot runs found", file=sys.stderr)
        return 1
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.validation_report
        if path.is_file()
    ]
    report = reports or None
    out = write_pilot_report(study, states, args.out, validation_report=report)
    text = out.read_text(encoding="utf-8")
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))
    go = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["gates"]["post_ln_go"]
    return {True: 0, False: 2, None: 3}[go]


if __name__ == "__main__":
    raise SystemExit(main())
