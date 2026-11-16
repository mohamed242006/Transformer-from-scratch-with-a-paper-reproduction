"""Write <run_dir>/REPORT.md (training summary + BLEU/chrF table) and print it.

python scripts/report.py --run-dir runs/multi30k_tiny_s1
"""

from __future__ import annotations

import argparse
import sys

from aiayn.inference import write_run_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    path = write_run_report(args.run_dir)
    text = path.read_text(encoding="utf-8")
    encoding = sys.stdout.encoding or "utf-8"  # e.g. cp1252 on Windows consoles
    print(text.encode(encoding, errors="replace").decode(encoding))
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
