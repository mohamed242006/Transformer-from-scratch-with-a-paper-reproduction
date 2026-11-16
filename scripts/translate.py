"""Translate sentences with a trained run (arguments, or one sentence per stdin line).

python scripts/translate.py --run-dir runs/multi30k_tiny_s1 "A man rides a bike."
cat sentences.txt | python scripts/translate.py --run-dir runs/x --checkpoint averaged
"""

from __future__ import annotations

import argparse
import sys

from aiayn.inference import DecodeConfig, Translator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("sentences", nargs="*")
    args = parser.parse_args(argv)

    sentences = args.sentences or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    translator = Translator.from_run(args.run_dir, checkpoint=args.checkpoint, device=args.device)
    config = DecodeConfig(beam_size=args.beam, length_penalty=args.alpha)
    for translation in translator.translate(sentences, config):
        print(translation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
