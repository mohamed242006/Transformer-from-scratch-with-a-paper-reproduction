"""Print training and validation curves of one or more runs from their files only.

Reads ``config.yaml``, ``metadata.json`` and ``metrics.jsonl`` (no checkpoints, works on
slimmed runs). For each run: the settings that matter for optimization, then one row per
window of updates with the learning rate, mean training NLL, the largest logged gradient
norm, the fp16 loss scale, cumulative skipped updates and the validation NLL. Flags mark
skipped-update bursts, gradient spikes and non-finite values (ADR-023 Phase 1b).

Usage:
    python scripts/training_curves.py runs/iwslt14_pilot_transformer_s1 \\
        runs/iwslt14_pilot_transformer-pre_ln_s1 [--every 500] [--csv curves.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml


def _get(tree: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(tree, dict):
            return None
        tree = tree.get(part)
    return tree


def load_run(run_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        (train if record.get("split") == "train" else valid).append(record)
    evals = {}
    if (run_dir / "eval").is_dir():
        for path in sorted((run_dir / "eval").glob("*.json")):
            evals[path.stem] = json.loads(path.read_text(encoding="utf-8")).get("bleu")
    return {"name": run_dir.name, "config": config, "meta": meta, "train": train,
            "valid": [r for r in valid if r.get("split") == "valid"], "evals": evals}  # fmt: skip


def windows(run: dict[str, Any], every: int) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for record in run["train"]:
        buckets.setdefault((int(record["step"]) - 1) // every, []).append(record)
    valid = {int(r["step"]): float(r["nll"]) for r in run["valid"]}
    grads = [float(r["grad_norm"]) for r in run["train"] if r.get("grad_norm") is not None]
    median_grad = statistics.median(grads) if grads else float("nan")
    rows, previous_skipped = [], 0
    for index in sorted(buckets):
        group = buckets[index]
        end = (index + 1) * every
        nlls = [float(r["nll"]) for r in group]
        group_grads = [float(r["grad_norm"]) for r in group if r.get("grad_norm") is not None]
        skipped = int(group[-1].get("skipped", 0))
        in_window = [s for s in valid if index * every < s <= end]
        flags = []
        if skipped - previous_skipped > max(2, len(group) // 10):
            flags.append(f"skipped +{skipped - previous_skipped}")
        if group_grads and median_grad and max(group_grads) > 10 * median_grad:
            flags.append("grad spike")
        if any(not math.isfinite(x) for x in nlls + group_grads):
            flags.append("NON-FINITE")
        rows.append(
            {
                "run": run["name"],
                "update": end,
                "lr": group[-1].get("lr"),
                "train_nll": sum(nlls) / len(nlls),
                "max_grad_norm": max(group_grads) if group_grads else None,
                "loss_scale": group[-1].get("loss_scale"),
                "skipped_total": skipped,
                "valid_nll": valid[max(in_window)] if in_window else None,
                "flags": "; ".join(flags),
            }
        )
        previous_skipped = skipped
    return rows


def _f(value: Any, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def describe(run: dict[str, Any]) -> str:
    c, m = run["config"], run["meta"]
    sessions = m.get("sessions", [])
    status = sessions[-1].get("status") if sessions else "?"
    items = [
        ("norm", _get(c, "model.norm_position") or _get(c, "model.attention")),
        ("layers", f"{_get(c, 'model.num_encoder_layers')}+{_get(c, 'model.num_decoder_layers')}"
         if _get(c, "model.num_encoder_layers") else _get(c, "model.num_layers")),
        ("peak_lr", _get(c, "train.schedule.peak_lr")),
        ("warmup", _get(c, "train.schedule.warmup_steps")),
        ("amp", _get(c, "train.amp")),
        ("clip", _get(c, "train.clip_norm")),
        ("adam_eps", _get(c, "train.optimizer.eps")),
        ("batch", f"{_get(c, 'data.batching.max_tokens')} x {_get(c, 'train.update_freq')}"),
        ("status", status),
        ("best_valid_nll", _f(m.get("best_valid_nll"), ".3f")),
        ("best_update", m.get("best_update")),
    ]  # fmt: skip
    evals = ", ".join(f"{k}={v}" for k, v in run["evals"].items())
    return " | ".join(f"{k}={v}" for k, v in items) + (f"\n  BLEU: {evals}" if evals else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--every", type=int, default=500, help="updates per row")
    parser.add_argument("--csv", type=Path, help="also write all rows to this CSV file")
    args = parser.parse_args(argv)

    all_rows = []
    for run_dir in args.runs:
        run = load_run(run_dir)
        rows = windows(run, args.every)
        all_rows += rows
        print(f"\n== {run['name']}\n  {describe(run)}")
        print(f"  {'update':>7} {'lr':>9} {'train NLL':>9} {'max grad':>9} {'loss scale':>10} "
              f"{'skipped':>7} {'valid NLL':>9}  flags")  # fmt: skip
        for r in rows:
            print(
                f"  {r['update']:>7} {_f(r['lr'], '.2e'):>9} {r['train_nll']:>9.3f} "
                f"{_f(r['max_grad_norm'], '.2f'):>9} {_f(r['loss_scale'], '.0f'):>10} "
                f"{r['skipped_total']:>7} {_f(r['valid_nll'], '.3f'):>9}  {r['flags']}"
            )
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
