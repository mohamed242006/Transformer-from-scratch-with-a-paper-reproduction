"""Compare trained runs under the fairness protocol (DESIGN.md §7.3, ADR-021).

Input: run directories (the first is the reference, e.g. the Transformer) that were all
evaluated with the same tag (default ``test.averaged.beam4``). Output: ``COMPARISON.md``.

The protocol is checked from the files, not assumed:

* identical data config, tokenizer fingerprint and reference lines;
* identical decoding settings and checkpoint choice (the shared tag);
* parameters within ±10% of the reference;
* every training-config difference listed; ``clip_norm`` is the one declared difference
  (an RNN stabilizer that can only help the baselines); anything else is flagged.

Significance: paired bootstrap resampling against the reference (sacreBLEU, 1000 samples).
With several seeds of one experiment, a mean ± std table is added.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiayn.evaluation.significance import PairedResult, paired_bootstrap
from aiayn.inference.evaluate import load_references
from aiayn.utils.config import load_yaml

__all__ = ["RunSummary", "build_comparison", "summarize_run", "write_comparison"]

PARAMETER_TOLERANCE = 0.10
#: Training keys that may differ between model families by protocol (ADR-021).
DECLARED_TRAIN_DIFFERENCES = frozenset({"clip_norm"})
#: Keys that never affect the result (run control, seeds are reported separately).
IGNORED_TRAIN_KEYS = frozenset(
    {"name", "seed", "time_budget_minutes", "log_every", "loggers", "device"}
)


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    name: str
    experiment: str
    description: str
    seed: int
    parameters: int
    best_valid_nll: float | None
    sessions_status: str
    wall_minutes: float
    tokens_per_s: float | None
    peak_mib: float | None
    config: dict[str, Any]
    tokenizer: str | None
    eval: dict[str, Any]
    hypotheses: list[str]


def _describe(model: dict[str, Any]) -> str:
    if model.get("architecture") == "rnn":
        prefix = "Bi" if model.get("bidirectional_encoder") else ""
        attention = {"none": "no attention", "bahdanau": "+ Bahdanau", "luong": "+ Luong"}
        return (
            f"{prefix}{str(model['cell']).upper()} {model['num_layers']}L "
            f"H={model['hidden_size']} {attention[model['attention']]}"
        )
    return (
        f"Transformer {model.get('norm_position', 'post')}-LN "
        f"{model['num_encoder_layers']}+{model['num_decoder_layers']}L d={model['d_model']}"
    )


def _wall_minutes(sessions: list[dict[str, Any]]) -> float:
    total = 0.0
    for s in sessions:
        if s.get("start") and s.get("end"):
            start, end = dt.datetime.fromisoformat(s["start"]), dt.datetime.fromisoformat(s["end"])
            total += (end - start).total_seconds() / 60
    return total


def _train_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, filter(None, lines)) if r.get("split") == "train"]


def summarize_run(run_dir: str | Path, tag: str = "test.averaged.beam4") -> RunSummary:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    config = load_yaml(run_dir / "config.yaml")
    eval_json = run_dir / "eval" / f"{tag}.json"
    if not eval_json.is_file():
        raise FileNotFoundError(f"{eval_json} missing: run scripts/evaluate.py first")
    hyps = (run_dir / "eval" / f"{tag}.hyp").read_text(encoding="utf-8").splitlines()
    train = _train_records(run_dir)
    speeds = [r["tgt_tokens_per_s"] for r in train if "tgt_tokens_per_s" in r]
    memory = [r["max_mem_mib"] for r in train if "max_mem_mib" in r]
    sessions = meta.get("sessions", [])
    return RunSummary(
        run_dir=run_dir,
        name=run_dir.name,
        experiment=str(meta.get("experiment", config.get("name"))),
        description=_describe(config["model"]),
        seed=int(config["train"].get("seed", 0)),
        parameters=int(meta.get("parameters", 0)),
        best_valid_nll=meta.get("best_valid_nll"),
        sessions_status=sessions[-1].get("status", "?") if sessions else "?",
        wall_minutes=_wall_minutes(sessions),
        tokens_per_s=statistics.median(speeds) if speeds else None,
        peak_mib=max(memory) if memory else None,
        config=config,
        tokenizer=meta.get("tokenizer_fingerprint"),
        eval=json.loads(eval_json.read_text(encoding="utf-8")),
        hypotheses=hyps,
    )


def _train_differences(reference: dict[str, Any], other: dict[str, Any]) -> list[str]:
    keys = sorted((set(reference) | set(other)) - IGNORED_TRAIN_KEYS)
    return [k for k in keys if reference.get(k) != other.get(k)]


def _fairness(runs: list[RunSummary], references: list[list[str]]) -> list[str]:
    ref = runs[0]
    lines = []

    def check(ok: bool, text: str) -> None:
        lines.append(f"- {'PASS' if ok else '**FAIL**'}: {text}")

    others = runs[1:]
    check(
        all(r.config["data"] == ref.config["data"] for r in others),
        "same data configuration (corpus, filters, splits, batching)",
    )
    check(all(r.tokenizer == ref.tokenizer for r in others), "same tokenizer fingerprint")
    check(all(refs == references[0] for refs in references[1:]), "same reference lines")
    check(
        all(
            (r.eval["decode"], r.eval["checkpoint"], r.eval.get("average_last"))
            == (ref.eval["decode"], ref.eval["checkpoint"], ref.eval.get("average_last"))
            for r in others
        ),
        f"same decoding and checkpoint ({ref.eval['checkpoint']}, {ref.eval['decode']})",
    )
    for r in others:
        ratio = r.parameters / ref.parameters - 1
        check(
            abs(ratio) <= PARAMETER_TOLERANCE,
            f"{r.name}: {r.parameters:,} parameters ({ratio:+.1%} vs reference, limit ±10%)",
        )
    for r in others:
        diffs = _train_differences(ref.config["train"], r.config["train"])
        undeclared = [k for k in diffs if k not in DECLARED_TRAIN_DIFFERENCES]
        detail = ", ".join(
            f"{k}: {ref.config['train'].get(k)} -> {r.config['train'].get(k)}" for k in diffs
        )
        check(
            not undeclared,
            f"{r.name}: training differences: {detail or 'none'}"
            + (f" (undeclared: {', '.join(undeclared)})" if undeclared else ""),
        )
    check(
        all(r.sessions_status == "completed" for r in runs),
        "all runs completed their training budget: "
        + ", ".join(f"{r.name}={r.sessions_status}" for r in runs),
    )
    return lines


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def build_comparison(
    run_dirs: Sequence[str | Path],
    *,
    tag: str = "test.averaged.beam4",
    significance: bool = True,
    n_samples: int = 1000,
    prepare_data: bool = False,
) -> str:
    """Markdown comparison of ``run_dirs`` (the first is the reference)."""
    if len(run_dirs) < 2:
        raise ValueError("need a reference run and at least one other run")
    runs = [summarize_run(d, tag) for d in run_dirs]
    split = runs[0].eval["split"]
    references = [load_references(r.run_dir, split, prepare_data=prepare_data) for r in runs]
    paired: list[PairedResult] | None = None
    signature = runs[0].eval.get("bleu_signature", "")
    if significance and all(refs == references[0] for refs in references[1:]):
        paired, signature = paired_bootstrap(
            [(r.name, r.hypotheses) for r in runs], references[0], n_samples=n_samples
        )

    ref = runs[0]
    lines = [
        f"# Comparison: {split} set, `{tag}`",
        "",
        f"Reference: `{ref.name}` ({ref.description}). Protocol: DESIGN.md §7.3, ADR-021.",
        "",
        "## Fairness checks",
        "",
        *_fairness(runs, references),
        "",
        "## Results",
        "",
        "| Run | Model | Params | Seed | Valid NLL (ppl) | **BLEU** | 95% CI | ΔBLEU | p "
        "| chrF | Len ratio | Wall (min) | tgt tok/s | Peak MiB | Decode (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(runs):
        e = r.eval
        nll = r.best_valid_nll
        ppl = None if nll is None else 2.718281828459045**nll
        ci = p = "-"
        if paired is not None:
            ci = f"±{paired[i].bleu_ci:.2f}"
            p = "ref" if i == 0 else _fmt(paired[i].bleu_p, ".4f")
        delta = "ref" if i == 0 else f"{e['bleu'] - ref.eval['bleu']:+.2f}"
        lines.append(
            f"| `{r.name}` | {r.description} | {r.parameters:,} | {r.seed} | "
            f"{_fmt(nll, '.3f')} ({_fmt(ppl, '.2f')}) | **{e['bleu']:.2f}** | {ci} | {delta} | "
            f"{p} | {e['chrf']:.2f} | {e['length_ratio']:.3f} | {r.wall_minutes:.1f} | "
            f"{_fmt(r.tokens_per_s, ',.0f')} | {_fmt(r.peak_mib, '.0f')} | "
            f"{e['decode_seconds']} |"
        )

    groups: dict[str, list[RunSummary]] = {}  # seeds of one experiment and model
    for r in runs:
        groups.setdefault(f"{r.experiment} / {r.config['model']['name']}", []).append(r)
    if any(len(g) > 1 for g in groups.values()):
        lines += [
            "",
            "## Across seeds (mean ± std)",
            "",
            "| Experiment / model | Seeds | BLEU | chrF | Valid NLL |",
            "|---|---|---|---|---|",
        ]
        for name, group in groups.items():

            def stat(values: list[float]) -> str:
                if len(values) < 2:
                    return f"{values[0]:.2f}"
                return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"

            nlls = [g.best_valid_nll for g in group if g.best_valid_nll is not None]
            lines.append(
                f"| `{name}` | {len(group)} | {stat([g.eval['bleu'] for g in group])} | "
                f"{stat([g.eval['chrf'] for g in group])} | {stat(nlls) if nlls else '-'} |"
            )

    lines += ["", "## Notes", ""]
    if paired is not None:
        lines.append(
            f"- 95% CI and p: paired bootstrap resampling ({n_samples} samples) against "
            f"`{ref.name}`; p is the probability of the observed difference under the null "
            "hypothesis that the system is no different from the reference."
        )
    else:
        lines.append("- Significance not computed (disabled or references differ).")
    lines += [
        "- Valid NLL: best teacher-forced validation NLL (unsmoothed), comparable because "
        "the tokenizer is identical. Wall: training sessions incl. validation and checkpoints.",
        "- Decoding re-runs the decoder over the prefix for every model (no KV cache or "
        "carried RNN state), so decode time compares implementations, not architectures.",
        f"- sacreBLEU: `{signature}`",
    ]
    return "\n".join(lines) + "\n"


def write_comparison(run_dirs: Sequence[str | Path], out: str | Path, **kwargs: Any) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_comparison(run_dirs, **kwargs), encoding="utf-8")
    return path
