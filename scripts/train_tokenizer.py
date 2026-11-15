"""Train the joint SentencePiece tokenizer on the training split (ADR-005).

Usage:
    python scripts/train_tokenizer.py --config configs/data/multi30k.yaml
"""

from __future__ import annotations

import argparse

from aiayn.data.config import load_data_config
from aiayn.data.datasets import read_lines
from aiayn.data.loading import load_tokenizer, train_tokenizer
from aiayn.data.prepare import data_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    config = load_data_config(args.config, tuple(args.override))
    if config.translation is None:
        print(f"{config.name}: synthetic task uses integer IDs; no tokenizer needed.")
        return 0

    model_path = train_tokenizer(config)
    tokenizer = load_tokenizer(config)
    print(f"Trained {model_path}")
    print(f"  vocab_size  {tokenizer.vocab_size}")
    print(f"  fingerprint {tokenizer.fingerprint}")

    sample = read_lines(data_paths(config).text_file("valid", config.translation.src_lang))[0]
    ids = tokenizer.encode(sample)
    print(f"  sample      {sample}")
    print(f"  pieces      {[tokenizer.id_to_piece(i) for i in ids]}")
    print(f"  round-trip  {tokenizer.decode(ids) == sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
