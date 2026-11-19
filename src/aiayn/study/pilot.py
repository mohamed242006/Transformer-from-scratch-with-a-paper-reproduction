"""Pilot report and go/no-go gates for the IWSLT'14 benchmark (ADR-023 §8).

Reads the pilot study's run directories (and, optionally, the Kaggle validation report)
and produces ``PILOT.md``: validation curves, throughput and memory, learning-rate
selection with grid-edge flags, gates G1-G4 with their evidence, and the resulting Phase 2
recommendation. Nothing here is hand-entered: every number comes from the run files.

Gates:

* **G1. Post-LN trains:** the default Post-LN Transformer completed, its best validation
  NLL is at the last evaluation, NLL decreased over the last three evaluations, and fewer
  than 1% of updates were skipped by the fp16 scaler.
* **G2. Post-LN is not worse than Pre-LN** (same learning rate): best validation NLL within
  +0.05 nats and validation BLEU within -0.5.
* **G3. Learning rate bracketed:** each model's selected rate is not at its grid's edge.
* **G4. Budget:** projected Phase 2 time per run fits one Kaggle session (informational
  beyond that: runs are resumable).

Post-LN is a GO when G1 and G2 pass; otherwise Phase 2 uses Pre-LN (ADR-023 amendment).
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiayn.study.config import StudyConfig
from aiayn.study.plan import (
    DEFAULT,
    default_valid,
    main_run,
    select_training,
    sensitivity_variants,
    training_edges,
    variant_run,
)
from aiayn.study.storage import RunState

__all__ = ["RunLog", "build_pilot_report", "read_log", "validation_report_summary"]

G2_NLL_MARGIN = 0.05
G2_BLEU_MARGIN = 0.5
SKIPPED_LIMIT = 0.01
SESSION_HOURS = 11.5
WEEKLY_QUOTA_HOURS = 30.0


@dataclass
class RunLog:
    name: str
    model: str
    variant: str
    completed: bool
    failed: bool
    valid: list[tuple[int, float]] = field(default_factory=list)  # (update, NLL)
    valid_bleu: float | None = None
    best_valid_nll: float | None = None
    final_train_nll: float | None = None
    tokens_per_s: float | None = None  # median over steady-state intervals
    peak_mib: float | None = None
    skipped: int = 0
    updates: int = 0
    max_grad_norm: float | None = None
    wall_minutes: float | None = None

    @property
    def minutes_per_1k(self) -> float | None:
        if not self.wall_minutes or not self.updates:
            return None
        return self.wall_minutes / self.updates * 1000


def _wall_minutes(meta: Mapping[str, Any]) -> float | None:
    total = 0.0
    for s in meta.get("sessions", []):
        if s.get("start") and s.get("end"):
            delta = dt.datetime.fromisoformat(s["end"]) - dt.datetime.fromisoformat(s["start"])
            total += delta.total_seconds() / 60
    return total or None


def read_log(state: RunState, *, name: str, model: str, variant: str, valid_tag: str) -> RunLog:
    log = RunLog(name, model, variant, state.completed, state.failed)
    log.best_valid_nll = state.best_valid_nll
    if valid_tag in state.evals:
        log.valid_bleu = float(state.evals[valid_tag]["bleu"])
    log.wall_minutes = _wall_minutes(state.metadata)
    path = state.path / "metrics.jsonl"
    if not path.is_file():
        return log
    train = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("split") == "valid":
            log.valid.append((int(record["step"]), float(record["nll"])))
        elif record.get("split") == "train":
            train.append(record)
    if train:
        log.final_train_nll = float(train[-1]["nll"])
        log.updates = int(train[-1]["step"])
        log.skipped = int(train[-1].get("skipped", 0))
        speeds = [float(r["tgt_tokens_per_s"]) for r in train if "tgt_tokens_per_s" in r]
        steady = speeds[2:] if len(speeds) > 4 else speeds  # first intervals include warm-up
        log.tokens_per_s = statistics.median(steady) if steady else None
        memory = [float(r["max_mem_mib"]) for r in train if "max_mem_mib" in r]
        log.peak_mib = max(memory) if memory else None
        grads = [float(r["grad_norm"]) for r in train if "grad_norm" in r]
        log.max_grad_norm = max(grads) if grads else None
    return log


def validation_report_summary(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per model: measured peak memory and throughput from ``kaggle_validate`` JSON."""
    out: dict[str, dict[str, Any]] = {}
    for check in report.get("checks", []):
        name = str(check.get("name", ""))
        if not name.startswith("model[") or "]." not in name:
            continue
        model, kind = name[len("model[") :].split("].", 1)
        details = check.get("details") or {}
        entry = out.setdefault(model, {})
        if kind == "memory":
            for key in (
                "peak_allocated_mib",
                "peak_reserved_mib",
                "device_total_mib",
                "estimated_peak_mib",
            ):
                if key in details:
                    entry[key] = details[key]
            entry["memory_status"] = check.get("status")
        elif kind == "throughput":
            entry["target_tokens_per_s"] = details.get("target_tokens_per_s")
            entry["ms_per_step"] = details.get("ms_per_step")
        elif kind == "build":
            entry["parameters"] = details.get("parameters")
    return out


def _f(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _status(ok: bool | None) -> str:
    return "PENDING" if ok is None else ("PASS" if ok else "**FAIL**")


class _Pilot:
    def __init__(self, study: StudyConfig, states: Mapping[str, RunState]) -> None:
        self.study = study
        self.states = states
        self.valid_tag = default_valid(study).tag
        self.logs: list[RunLog] = []
        for model in study.models:
            self._add(main_run(study, model, study.main_seeds[0]).name, model, DEFAULT)
            for v in sensitivity_variants(study, model):
                name = variant_run(study, model, "sensitivity", v, study.sensitivity.seeds[0]).name
                self._add(name, model, v.name)
            for v in study.ablations.variants:
                if model in study.ablations.models and v.applies_to(model):
                    name = variant_run(study, model, "ablation", v, study.ablations.seeds[0]).name
                    self._add(name, model, v.name)

    def _add(self, name: str, model: str, variant: str) -> None:
        state = self.states.get(name)
        if state is None:
            self.logs.append(RunLog(name, model, variant, False, False))
        else:
            self.logs.append(
                read_log(state, name=name, model=model, variant=variant, valid_tag=self.valid_tag)
            )

    def log(self, model: str, variant: str) -> RunLog | None:
        return next((g for g in self.logs if g.model == model and g.variant == variant), None)


def _lr_of(study: StudyConfig, model: str, variant: str) -> str:
    if variant == DEFAULT:
        from aiayn.training.experiment import load_experiment

        return f"{load_experiment(study.models[model]).train.schedule.peak_lr:g}"
    for v in study.sensitivity.variants:
        if v.name == variant:
            for item in v.overrides:
                if item.startswith("train.schedule.peak_lr="):
                    return f"{float(item.split('=', 1)[1]):g}"
    return "-"


def build_pilot_report(
    study: StudyConfig,
    states: Mapping[str, RunState],
    *,
    transformer: str = "transformer",
    pre_ln_variant: str = "pre_ln",
    validation_report: Mapping[str, Any] | None = None,
    phase2_updates: int = 30000,
    phase2_seeds: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Markdown report + machine-readable data (also used to write the curves CSV)."""
    pilot = _Pilot(study, states)
    done = sum(g.completed and not g.failed for g in pilot.logs)
    failed = [g.name for g in pilot.logs if g.failed]
    missing = [g.name for g in pilot.logs if not g.completed and not g.failed]
    lines = [
        f"# Pilot report: `{study.name}`",
        "",
        "Gates and thresholds: ADR-023 §8. Every number is read from the run files.",
        "",
        f"**Status:** {done}/{len(pilot.logs)} runs completed"
        + (f", failed: {', '.join(failed)}" if failed else "")
        + (f", not finished: {', '.join(missing)}" if missing else "")
        + ".",
    ]
    data: dict[str, Any] = {
        "runs": [vars(g) | {"minutes_per_1k": g.minutes_per_1k} for g in pilot.logs]
    }

    # 1. Validation curves
    steps = sorted({s for g in pilot.logs for s, _ in g.valid})
    labels = [f"{g.model}:{g.variant}" for g in pilot.logs]
    lines += [
        "",
        "## 1. Validation curves (teacher-forced NLL)",
        "",
        "| Update | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    curves: list[dict[str, Any]] = []
    for step in steps:
        row: dict[str, Any] = {"update": step}
        cells = []
        for g, label in zip(pilot.logs, labels, strict=True):
            value = dict(g.valid).get(step)
            row[label] = value
            cells.append(_f(value))
        curves.append(row)
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    data["curves"] = curves

    # 2. Throughput, memory, stability, projection
    lines += [
        "",
        "## 2. Throughput, memory and stability (T4 training logs)",
        "",
        "| Run | Peak LR | Target tok/s | Peak MiB | Skipped fp16 | Max grad norm "
        "| Final train NLL | Best valid NLL | Valid BLEU | Min per 1k updates | Phase 2 run (h) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in pilot.logs:
        projected = g.minutes_per_1k * phase2_updates / 1000 / 60 if g.minutes_per_1k else None
        lines.append(
            f"| `{g.name}` | {_lr_of(study, g.model, g.variant)} | {_f(g.tokens_per_s, ',.0f')} | "
            f"{_f(g.peak_mib, '.0f')} | {g.skipped} | {_f(g.max_grad_norm, '.2f')} | "
            f"{_f(g.final_train_nll)} | {_f(g.best_valid_nll)} | {_f(g.valid_bleu, '.2f')} | "
            f"{_f(g.minutes_per_1k, '.1f')} | {_f(projected, '.1f')} |"
        )
    if validation_report:
        summary = validation_report_summary(validation_report)
        data["validation_report"] = summary
        lines += [
            "",
            "Kaggle validation (worst-case batch at the data config's token budget):",
            "",
            "| Model | Parameters | Peak allocated MiB | Peak reserved MiB | Device MiB "
            "| Target tok/s | ms/step |",
            "|---|---|---|---|---|---|---|",
        ]
        for model, e in summary.items():
            lines.append(
                f"| {model} | {_f(e.get('parameters'), ',')} | "
                f"{_f(e.get('peak_allocated_mib'), '.0f')} | "
                f"{_f(e.get('peak_reserved_mib'), '.0f')} | "
                f"{_f(e.get('device_total_mib'), '.0f')} | "
                f"{_f(e.get('target_tokens_per_s'), ',.0f')} | "
                f"{_f(e.get('ms_per_step'), '.1f')} |"
            )

    # 3. Learning-rate selection (G3)
    lines += ["", "## 3. Learning-rate selection (validation BLEU, NLL breaks ties)", ""]
    selections: dict[str, Any] = {}
    g3_ok: bool | None = True
    for model in study.models:
        choice = select_training(study, model, states)
        edges = training_edges(study, model, choice) if choice else []
        if choice is None:
            g3_ok = None if g3_ok is not False else g3_ok
        elif edges:
            g3_ok = False
        candidates = [g for g in pilot.logs if g.model == model and g.variant != "pre_ln"]
        ranked = ", ".join(
            f"{_lr_of(study, model, g.variant)}: BLEU {_f(g.valid_bleu, '.2f')} / "
            f"NLL {_f(g.best_valid_nll)}"
            for g in candidates
        )
        selections[model] = {
            "variant": choice,
            "peak_lr": _lr_of(study, model, choice) if choice else None,
            "edges": edges,
        }
        flag = (
            f" ⚠ {'; '.join(edges)}: add one pilot run beyond the edge before Phase 2"
            if edges
            else ""
        )
        lines.append(
            f"- **{model}**: selected `{choice or 'pending'}`"
            f" (peak LR {selections[model]['peak_lr'] or '-'}){flag}. Candidates: {ranked}."
        )
    data["selection"] = selections

    # 4. Gates
    post = pilot.log(transformer, DEFAULT)
    pre = pilot.log(transformer, pre_ln_variant)
    g1_ok: bool | None = None
    g1_evidence = "default Post-LN run not finished"
    if post is not None and (post.completed or post.failed):
        nlls = [n for _, n in post.valid]
        best_last = bool(nlls) and nlls[-1] == min(nlls)
        improving = len(nlls) >= 3 and nlls[-3] > nlls[-2] > nlls[-1]
        skipped_ok = post.updates > 0 and post.skipped / post.updates < SKIPPED_LIMIT
        g1_ok = post.completed and not post.failed and best_last and improving and skipped_ok
        g1_evidence = (
            f"completed={post.completed and not post.failed}, best at last evaluation={best_last}, "
            f"decreasing over last 3={improving} ({', '.join(_f(n) for n in nlls[-3:])}), "
            f"skipped {post.skipped}/{post.updates}"
        )
    g2_ok: bool | None = None
    g2_evidence = "Post-LN or Pre-LN run not finished"
    if post and pre and post.best_valid_nll is not None and pre.best_valid_nll is not None:
        nll_ok = post.best_valid_nll <= pre.best_valid_nll + G2_NLL_MARGIN
        bleu_ok = (
            post.valid_bleu is not None
            and pre.valid_bleu is not None
            and post.valid_bleu >= pre.valid_bleu - G2_BLEU_MARGIN
        )
        g2_ok = nll_ok and bleu_ok
        g2_evidence = (
            f"best valid NLL Post {_f(post.best_valid_nll)} vs Pre {_f(pre.best_valid_nll)} "
            f"(limit +{G2_NLL_MARGIN}); valid BLEU Post {_f(post.valid_bleu, '.2f')} vs Pre "
            f"{_f(pre.valid_bleu, '.2f')} (limit -{G2_BLEU_MARGIN})"
        )
    projections = {}
    for model, sel in selections.items():
        run = pilot.log(model, sel["variant"]) if sel["variant"] else None
        if run is not None and run.minutes_per_1k:
            projections[model] = run.minutes_per_1k * phase2_updates / 1000 / 60
    total = sum(projections.values()) * phase2_seeds
    g4_ok: bool | None = None
    if len(projections) == len(study.models):
        g4_ok = max(projections.values()) <= SESSION_HOURS
    g4_evidence = (
        ", ".join(f"{m} {h:.1f} h/run" for m, h in projections.items())
        + f"; total {total:.1f} h for {phase2_seeds} seeds "
        f"(~{total / WEEKLY_QUOTA_HOURS:.1f} weekly quotas of {WEEKLY_QUOTA_HOURS:g} h)"
        if projections
        else "pending"
    )
    go = None if g1_ok is None or g2_ok is None else (g1_ok and g2_ok)
    lines += [
        "",
        "## 4. Gates",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
        f"| G1 Post-LN trains | {_status(g1_ok)} | {g1_evidence} |",
        f"| G2 Post-LN not worse than Pre-LN | {_status(g2_ok)} | {g2_evidence} |",
        f"| G3 learning rates bracketed | {_status(g3_ok)} | see section 3 |",
        f"| G4 Phase 2 run fits a session | {_status(g4_ok)} | {g4_evidence} |",
        "",
        "**Post-LN decision:** "
        + {
            None: "PENDING",
            True: "**GO**: Phase 2 headline Transformer stays Post-LN.",
            False: "**NO-GO**: Phase 2 headline Transformer switches to Pre-LN "
            "(ADR-023 amendment).",
        }[go],
    ]

    # 5. Recommendation
    rec = ["", "## 5. Phase 2 recommendation", ""]
    if go is None or any(s["variant"] is None for s in selections.values()):
        rec.append("_Pending: finish the pilot first._")
    else:
        norm = "post" if go else "pre"
        rec.append(f"- Transformer: `model.norm_position={norm}`.")
        for model, sel in selections.items():
            extra = (
                " **First run one more pilot point beyond the grid edge**; "
                "the optimum is not bracketed."
                if sel["edges"]
                else ""
            )
            rec.append(f"- {model}: `train.schedule.peak_lr={sel['peak_lr']}`.{extra}")
        rec.append(
            f"- {phase2_seeds} seeds per model, 30k updates, valid-selected checkpoint and "
            f"decoding (ADR-023 §7); projected {total:.1f} T4-hours in total."
        )
        if g4_ok is False:
            rec.append("- Some runs exceed one session: they resume across sessions (ADR-013).")
    lines += rec
    data["gates"] = {
        "G1": {"ok": g1_ok, "evidence": g1_evidence},
        "G2": {"ok": g2_ok, "evidence": g2_evidence},
        "G3": {"ok": g3_ok},
        "G4": {"ok": g4_ok, "evidence": g4_evidence, "projected_hours": projections},
        "post_ln_go": go,
    }
    return "\n".join(lines) + "\n", data


def write_pilot_report(
    study: StudyConfig,
    states: Mapping[str, RunState],
    out: Path,
    *,
    validation_report: Mapping[str, Any] | None = None,
) -> Path:
    text, data = build_pilot_report(study, states, validation_report=validation_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    curves = data["curves"]
    if curves:
        columns: Sequence[str] = list(curves[0])
        rows = [",".join(columns)] + [
            ",".join("" if row.get(c) is None else str(row.get(c)) for c in columns)
            for row in curves
        ]
        out.with_name("pilot_curves.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out
