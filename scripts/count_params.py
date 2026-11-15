"""Build a model from a config and compare counted vs analytic parameters (math.md §9).

Usage:
    python scripts/count_params.py --config configs/model/transformer_base.yaml --vocab-size 37000
"""

from __future__ import annotations

import argparse

from aiayn.data.special_tokens import PAD_ID
from aiayn.models import build_transformer, expected_parameter_count, load_model_config
from aiayn.utils.params import count_parameters, parameter_breakdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    config = load_model_config(args.config, tuple(args.override))
    model = build_transformer(config, vocab_size=args.vocab_size, pad_id=PAD_ID, seed=0)
    counted = count_parameters(model)
    expected = expected_parameter_count(config, args.vocab_size)
    print(f"{config.name} (V={args.vocab_size:,})")
    for name, n in parameter_breakdown(model).items():
        print(f"  {name:<16} {n:>13,}")
    print(f"  {'total':<16} {counted:>13,}   analytic {expected:,}   match={counted == expected}")
    return 0 if counted == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
