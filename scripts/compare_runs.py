"""Compare evaluated runs under the fairness protocol (ADR-021) and write COMPARISON.md.

The first ``--run`` is the reference (the Transformer). Every run must have been
evaluated with the same tag (default ``test.averaged.beam4``, see scripts/evaluate.py).

Usage:
    python scripts/compare_runs.py \\
        --run runs/multi30k_tiny_s1 --run runs/multi30k_bilstm_luong_s1 \\
        --run runs/multi30k_gru_bahdanau_s1 --run runs/multi30k_lstm_seq2seq_s1 \\
        --out runs/COMPARISON.md

Exit codes: 0 = written and every fairness check passed; 1 = written with a failed check.
"""

from __future__ import annotations

import argparse
import sys

from aiayn.inference.comparison import build_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="run directory (repeat)")
    parser.add_argument("--tag", default="test.averaged.beam4", help="eval/<tag>.json to compare")
    parser.add_argument("--out", default=None, help="write the markdown here (default: print)")
    parser.add_argument("--samples", type=int, default=1000, help="bootstrap samples")
    parser.add_argument("--no-significance", action="store_true")
    parser.add_argument("--prepare-data", action="store_true", help="download data if missing")
    args = parser.parse_args(argv)

    text = build_comparison(
        args.run,
        tag=args.tag,
        significance=not args.no_significance,
        n_samples=args.samples,
        prepare_data=args.prepare_data,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))
    return 1 if "**FAIL**" in text else 0


if __name__ == "__main__":
    raise SystemExit(main())
