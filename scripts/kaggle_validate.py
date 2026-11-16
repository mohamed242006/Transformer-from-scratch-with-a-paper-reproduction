"""Validate a (Kaggle) environment end to end: environment, dependencies, CUDA, data,
tokenizer, model construction, forward/backward, VRAM and throughput.

Exit code 0 if no critical check failed, 1 otherwise. A JSON report is always written.

Examples:
    # Kaggle (GPU, offline data from attached datasets)
    python scripts/kaggle_validate.py \\
        --data-config configs/data/multi30k.yaml \\
        --model-config configs/model/transformer_tiny.yaml \\
        --model-config configs/model/transformer_iwslt.yaml \\
        --data-override translation.root=/kaggle/working/data \\
        --data-mirror /kaggle/input/aiayn-multi30k \\
        --tokenizer /kaggle/input/aiayn-multi30k/multi30k_bpe8000.model \\
        --expected-fingerprint-file /kaggle/input/aiayn-multi30k/tokenizer_fingerprint.txt \\
        --require-cuda --output /kaggle/working/validation/report.json

    # Local CPU smoke test (VRAM is estimated, not measured)
    python scripts/kaggle_validate.py --data-config configs/data/multi30k.yaml \\
        --model-config configs/model/transformer_tiny.yaml --device cpu --steps 2 --warmup 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aiayn.diagnostics.validation import ValidationOptions, run_validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, action="append", required=True)
    parser.add_argument("--data-override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-mirror", type=Path, action="append", default=[])
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--expected-fingerprint-file", type=Path)
    parser.add_argument("--train-tokenizer-if-missing", action="store_true")
    parser.add_argument(
        "--retrain-tokenizer-check",
        action="store_true",
        help="retrain the tokenizer and require an identical fingerprint",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", choices=["auto", "fp16", "off"], default="auto")
    parser.add_argument(
        "--vram-max-tokens",
        type=int,
        help="token budget for the worst-case memory batch (default: data config)",
    )
    parser.add_argument("--steps", type=int, default=30, help="timed throughput steps")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--expected-torch-version", help="torch.__version__ recorded before installing dependencies"
    )
    parser.add_argument("--no-internet-check", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("validation_report.json"))
    args = parser.parse_args(argv)

    fingerprint = args.expected_fingerprint
    if args.expected_fingerprint_file is not None:
        fingerprint = args.expected_fingerprint_file.read_text(encoding="utf-8").strip()

    options = ValidationOptions(
        data_config=args.data_config,
        model_configs=args.model_config,
        data_overrides=tuple(args.data_override),
        data_mirrors=args.data_mirror,
        tokenizer_path=args.tokenizer,
        expected_fingerprint=fingerprint,
        train_tokenizer_if_missing=args.train_tokenizer_if_missing,
        retrain_tokenizer_check=args.retrain_tokenizer_check,
        device=args.device,
        require_cuda=args.require_cuda,
        amp=args.amp,
        vram_max_tokens=args.vram_max_tokens,
        steps=args.steps,
        warmup=args.warmup,
        expected_torch_version=args.expected_torch_version,
        check_internet=not args.no_internet_check,
    )
    report = run_validation(options, output_dir=args.output.parent)
    path = report.write_json(args.output)
    print(report.format_table())
    print(f"report: {path}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
