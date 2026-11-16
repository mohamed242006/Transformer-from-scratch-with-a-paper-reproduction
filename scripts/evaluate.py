"""Decode a split of a run's dataset and score it with sacreBLEU and chrF (ADR-006, ADR-019).

Examples:
    python scripts/evaluate.py --run-dir runs/multi30k_tiny_s1   # test split, best, beam 4
    python scripts/evaluate.py --run-dir runs/multi30k_tiny_s1 --checkpoint averaged
    python scripts/evaluate.py --run-dir runs/multi30k_tiny_s1 --greedy --split valid

Writes runs/<run>/eval/<split>.<checkpoint>.<search>.{hyp,json}.
"""

from __future__ import annotations

import argparse
import json

from aiayn.inference import DecodeConfig, evaluate_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--checkpoint", default="best", help="best | last | averaged | <path>")
    parser.add_argument("--average-last", type=int, help="snapshots to average (default: all)")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--greedy", action="store_true", help="same as --beam 1")
    parser.add_argument("--alpha", type=float, default=0.6, help="GNMT length penalty")
    parser.add_argument("--max-len-a", type=float, default=1.0)
    parser.add_argument("--max-len-b", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=4096, help="beam x tokens per batch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prepare-data", action="store_true")
    args = parser.parse_args(argv)

    decode = DecodeConfig(
        beam_size=1 if args.greedy else args.beam,
        length_penalty=args.alpha,
        max_len_a=args.max_len_a,
        max_len_b=args.max_len_b,
    )
    result = evaluate_run(
        args.run_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        decode=decode,
        average_last=args.average_last,
        device=args.device,
        max_tokens=args.max_tokens,
        prepare_data=args.prepare_data,
    )
    keys = (
        "split",
        "checkpoint",
        "bleu",
        "bleu_lowercase",
        "chrf",
        "exact_match",
        "length_ratio",
        "sentences",
        "decode_seconds",
    )
    print(json.dumps({k: result[k] for k in keys}, indent=1))
    print("sacreBLEU:", result["bleu_signature"])
    print("by source length:", result["bleu_by_length"])
    print("written:", result.get("output"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
