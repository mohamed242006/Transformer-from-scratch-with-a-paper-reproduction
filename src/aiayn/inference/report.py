"""Human-readable run report (``REPORT.md``): training summary + BLEU/chrF results.

Built only from files in the run directory (``metadata.json``, ``metrics.jsonl``,
``eval/*.json``), so it can be regenerated anywhere, including from a downloaded run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["build_run_report", "write_run_report"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in records if r.get("split") == "valid"]


def build_run_report(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "metadata.json")
    lines = [f"# Run report: `{run_dir.name}`", ""]
    lines += [
        "| | |",
        "|---|---|",
        f"| Experiment | `{meta.get('experiment')}` |",
        f"| Parameters | {meta.get('parameters', 0):,} |",
        f"| Config hash | `{str(meta.get('config_hash'))[:16]}` |",
        f"| Code tree | `{str(meta.get('code_tree_sha256'))[:16]}` |",
        f"| Tokenizer | `{str(meta.get('tokenizer_fingerprint'))[:16]}` |",
        f"| Best validation NLL | {meta.get('best_valid_nll')} "
        f"(update {meta.get('best_update')}) |",
        "",
        "## Sessions",
        "",
        "| # | Start | Status | Updates | Device | AMP |",
        "|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(meta.get("sessions", []), start=1):
        lines.append(
            f"| {i} | {s.get('start')} | {s.get('status')} | "
            f"{s.get('start_update')} -> {s.get('end_update')} | "
            f"{s.get('device')} | {s.get('amp')} |"
        )

    valid = _validation_records(run_dir)
    if valid:
        last = valid[-1]
        lines += [
            "",
            "## Validation (teacher-forced)",
            "",
            f"{len(valid)} evaluations; last at update {last['step']}: "
            f"NLL {last.get('nll'):.4f}, perplexity {last.get('ppl'):.2f}, "
            f"token accuracy {last.get('accuracy'):.4f}.",
        ]

    results = sorted((run_dir / "eval").glob("*.json")) if (run_dir / "eval").is_dir() else []
    lines += ["", "## Translation quality", ""]
    if not results:
        lines.append("_No evaluation yet: run `scripts/evaluate.py --run-dir <run>`._")
    else:
        lines += [
            "| Split | Checkpoint | Search | **BLEU** | lc-BLEU | chrF | Exact | Len ratio "
            "| Sents | Time (s) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        evals = [_read_json(p) for p in results]
        for r in evals:
            beam = r["decode"]["beam_size"]
            search = (
                "greedy" if beam == 1 else f"beam {beam}, alpha={r['decode']['length_penalty']}"
            )
            lines.append(
                f"| {r['split']} | {r['checkpoint']} | {search} | **{r['bleu']:.2f}** | "
                f"{r['bleu_lowercase']:.2f} | {r['chrf']:.2f} | {r['exact_match']:.3f} | "
                f"{r['length_ratio']:.3f} | {r['sentences']} | {r['decode_seconds']} |"
            )
        best = max(evals, key=lambda r: r["bleu"])
        lines += ["", f"BLEU by source length ({best['split']}, {best['checkpoint']}):", ""]
        lines += ["| Bucket (words) | Sentences | BLEU |", "|---|---|---|"]
        lines += [
            f"| {b['bucket']} | {b['sentences']} | {b['bleu']} |" for b in best["bleu_by_length"]
        ]
        lines += ["", f"sacreBLEU signature: `{best['bleu_signature']}`"]
    return "\n".join(lines) + "\n"


def write_run_report(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "REPORT.md"
    path.write_text(build_run_report(run_dir), encoding="utf-8")
    return path
