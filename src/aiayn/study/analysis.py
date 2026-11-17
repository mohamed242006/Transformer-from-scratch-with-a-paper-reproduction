"""Study report: tables, tests and notes from run files only (ADR-022).

Sections:
1. Main comparison (protocol defaults, all main seeds).
2. Comparison with valid-selected decoding.
3. Comparison with the valid-selected training configuration ("tuned").
4. Training sensitivity (learning rate, dropout).
5. Decoding sensitivity (beam × α grid on valid).
6. Checkpoint averaging and beam search (paired, same runs).
7. Transformer ablations (paper Table 3 style).
8. Statistical notes and fairness checks.

Primary test: two-level paired bootstrap (sentences × seeds) of the mean-over-seeds BLEU
difference, Holm-adjusted within each table. Supplementary: exact seed permutation test.
Missing runs are reported, never silently dropped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aiayn.evaluation.statistics import (
    BleuBootstrap,
    compare_bleu,
    format_p,
    holm,
    mean_std,
    permutation_test,
    spearman,
    t_halfwidth,
)
from aiayn.study.config import StudyConfig
from aiayn.study.plan import (
    DEFAULT,
    EvalSpec,
    all_run_names,
    default_test,
    grid_evals,
    is_alias,
    main_run,
    planned_runs,
    select_decoding,
    select_training,
    tasks,
    variant_run,
)
from aiayn.study.storage import RunState
from aiayn.utils.config import load_yaml

__all__ = ["System", "build_study_report"]


@dataclass(frozen=True)
class System:
    """One configuration evaluated under one tag, over its available seeds."""

    label: str
    runs: tuple[RunState, ...]
    tag: str

    @property
    def scored(self) -> tuple[RunState, ...]:
        return tuple(r for r in self.runs if r.usable and self.tag in r.evals)

    def values(self, key: str) -> list[float]:
        return [float(r.evals[self.tag][key]) for r in self.scored]

    def nlls(self) -> list[float]:
        return [float(r.best_valid_nll) for r in self.scored if r.best_valid_nll is not None]

    def hypotheses(self) -> list[list[str]]:
        return [
            (r.path / "eval" / f"{self.tag}.hyp").read_text(encoding="utf-8").splitlines()
            for r in self.scored
        ]


def _ms(values: Sequence[float], digits: int = 2) -> str:
    if not values:
        return "-"
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if len(values) > 1 else f"{mean:.{digits}f}"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _f(value: float | None, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def _delta(value: float | None, base: float | None, is_base: bool) -> str:
    return "-" if value is None or base is None or is_base else f"{value - base:+.2f}"


def _decode_label(choice: EvalSpec | None) -> str:
    return "pending" if choice is None else f"beam {choice.beam}, α {choice.alpha:g}"


def _ci(test: Mapping[str, Any]) -> str:
    return f"[{test['ci'][0]:+.2f}, {test['ci'][1]:+.2f}]"


def _check(ok: bool) -> str:
    return "PASS" if ok else "**FAIL**"


class Analyzer:
    def __init__(
        self, study: StudyConfig, states: Mapping[str, RunState], references: Sequence[str]
    ):
        self.study = study
        self.states = states
        self.bootstrap = BleuBootstrap(references, study.bootstrap_samples)
        self.n = study.bootstrap_samples
        self.data: dict[str, Any] = {}
        self._seed = 0

    # ------------------------------------------------------------------ helpers
    def runs(self, names: Sequence[str]) -> tuple[RunState, ...]:
        return tuple(self.states[n] for n in names if n in self.states)

    def main_system(self, model: str, tag: str | None = None) -> System:
        names = [main_run(self.study, model, s).name for s in self.study.main_seeds]
        return System(model, self.runs(names), tag or default_test(self.study).tag)

    def variant_system(
        self, model: str, variant_name: str, seeds: Sequence[int], label: str
    ) -> System:
        if variant_name == DEFAULT:
            names = [main_run(self.study, model, s).name for s in seeds]
        else:
            variant = next(
                v
                for g in (self.study.sensitivity, self.study.ablations)
                for v in g.variants
                if v.name == variant_name
            )
            names = [variant_run(self.study, model, "sensitivity", variant, s).name for s in seeds]
        return System(label, self.runs(names), default_test(self.study).tag)

    def compare(self, a: System, b: System) -> dict[str, Any] | None:
        if not a.scored or not b.scored:
            return None
        self._seed += 1
        boot = compare_bleu(self.bootstrap, a.hypotheses(), b.hypotheses(), seed=self._seed)
        perm = permutation_test(a.values("bleu"), b.values("bleu"))
        return {
            "delta": boot.delta,
            "ci": [boot.ci_low, boot.ci_high],
            "p_bootstrap": boot.p_value,
            "p_permutation": perm,
        }

    # ------------------------------------------------------------------ tables
    def comparison_table(self, key: str, systems: Sequence[System], reference: System) -> list[str]:
        rows, entries = [], []
        is_ref = [s is reference or s.label == reference.label for s in systems]
        tests = [
            None if r else self.compare(s, reference) for s, r in zip(systems, is_ref, strict=True)
        ]
        adjusted = holm([t["p_bootstrap"] if t else None for t in tests])
        lines = [
            "| Model | Params | Seeds | **BLEU** mean ± std | 95% CI (seeds) | min–max | chrF "
            "| Len ratio | Valid NLL | ΔBLEU | 95% CI (Δ) | p | p (Holm) | p (seed perm.) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for system, test, p_adj, ref in zip(systems, tests, adjusted, is_ref, strict=True):
            bleu = system.values("bleu")
            params = system.scored[0].parameters if system.scored else None
            entry = {
                "label": system.label,
                "tag": system.tag,
                "seeds": len(bleu),
                "runs": [r.path.name for r in system.scored],
                "bleu": bleu,
                "chrf": system.values("chrf"),
                "length_ratio": system.values("length_ratio"),
                "valid_nll": system.nlls(),
                "test": test,
                "p_holm": p_adj,
            }
            entries.append(entry)
            if ref:
                delta = ci = p = ph = pp = "ref"
            elif test is None:
                delta = ci = p = ph = pp = "-"
            else:
                delta = f"{test['delta']:+.2f}"
                ci = f"[{test['ci'][0]:+.2f}, {test['ci'][1]:+.2f}]"
                p, ph = format_p(test["p_bootstrap"], self.n), format_p(p_adj, self.n)
                pp = f"{test['p_permutation']:.3f}"
            seed_ci = "±" + _f(t_halfwidth(bleu)) if len(bleu) > 1 else "-"
            span = f"{min(bleu):.2f}–{max(bleu):.2f}" if bleu else "-"
            ratio = _ms(system.values("length_ratio"), 3)
            rows.append(
                f"| {system.label} | {_f(params, ',') if params else '-'} | {len(bleu)} | "
                f"**{_ms(bleu)}** | {seed_ci} | {span} | {_ms(system.values('chrf'))} | "
                f"{ratio} | {_ms(system.nlls(), 3)} | {delta} | {ci} | {p} | {ph} | {pp} |"
            )
        self.data[key] = entries
        return lines + rows

    def section_main(self) -> list[str]:
        reference = self.main_system(self.study.reference)
        systems = [self.main_system(m) for m in self.study.models]
        return [
            "## 1. Main comparison (protocol defaults)",
            "",
            f"Test set, averaged checkpoint, beam {self.study.default_beam}, "
            f"α = {self.study.default_alpha}; seeds {list(self.study.main_seeds)}.",
            "",
            *self.comparison_table("main", systems, reference),
        ]

    def section_decoding_selected(self) -> list[str]:
        lines = ["## 2. Comparison with decoding selected on validation", ""]
        selected: dict[str, EvalSpec] = {}
        for model in self.study.models:
            choice = select_decoding(self.study, model, self.states)
            if choice is not None:
                selected[model] = choice
        if len(selected) < len(self.study.models):
            return [*lines, "_Pending: the decoding grid is not complete for every model._"]
        lines.append(
            "Each model decodes with its own valid-best (beam, α) (section 5); "
            "everything else as in section 1."
        )
        lines.append("")
        systems = [
            System(
                f"{m} ({selected[m].tag.split('.')[-1]})", self.main_system(m).runs, selected[m].tag
            )
            for m in self.study.models
        ]
        reference = systems[list(self.study.models).index(self.study.reference)]
        self.data["decoding_selection"] = {m: [s.beam, s.alpha] for m, s in selected.items()}
        return [*lines, *self.comparison_table("decoding_selected", systems, reference)]

    def section_tuned(self) -> list[str]:
        lines = ["## 3. Comparison with training configuration selected on validation", ""]
        choices = {m: select_training(self.study, m, self.states) for m in self.study.models}
        if any(c is None for c in choices.values()):
            return [*lines, "_Pending: the sensitivity grid is not complete for every model._"]
        lines += [
            "Per model, the sensitivity variant with the lowest mean best-validation NLL "
            f"(seeds {list(self.study.sensitivity.seeds)}), run on all main seeds:",
            "",
        ]
        lines += [f"- {m}: `{c}`" for m, c in choices.items()]
        lines.append("")
        systems = [
            self.variant_system(m, str(choices[m]), self.study.main_seeds, f"{m} ({choices[m]})")
            for m in self.study.models
        ]
        reference = systems[list(self.study.models).index(self.study.reference)]
        self.data["training_selection"] = choices
        return [*lines, *self.comparison_table("tuned", systems, reference)]

    def section_sensitivity(self) -> list[str]:
        group = self.study.sensitivity
        lines = [
            "## 4. Training sensitivity",
            "",
            f"Seeds {list(group.seeds)}; test BLEU at protocol decoding. Same grid for every model "
            "(DESIGN.md §7.3). `= default` marks a variant equal to the model's default config.",
        ]
        out: dict[str, Any] = {}
        for model in group.models:
            default = self.variant_system(model, DEFAULT, group.seeds, "default")
            base = _mean(default.values("bleu"))
            choice = select_training(self.study, model, self.states)
            lines += [
                "",
                f"**{model}** (selected: `{choice or 'pending'}`)",
                "",
                "| Variant | Overrides | Done | Failed | Valid NLL | Test BLEU | Δ vs default |",
                "|---|---|---|---|---|---|---|",
            ]
            rows: dict[str, Any] = {}
            for name, overrides, system in [
                ("default", "", default),
                *[
                    (
                        v.name,
                        " ".join(f"`{o}`" for o in v.overrides),
                        None
                        if is_alias(self.study, model, v)
                        else self.variant_system(model, v.name, group.seeds, v.name),
                    )
                    for v in group.variants
                ],
            ]:
                if system is None:
                    lines.append(f"| {name} | {overrides} | = default | | | | |")
                    continue
                failed = sum(r.failed for r in system.runs)
                bleu = system.values("bleu")
                delta = _mean(bleu)
                lines.append(
                    f"| {name} | {overrides} | {len(bleu)}/{len(group.seeds)} | {failed or ''} | "
                    f"{_ms(system.nlls(), 3)} | {_ms(bleu)} | "
                    f"{_delta(delta, base, name == 'default')} |"
                )
                rows[name] = {"bleu": bleu, "valid_nll": system.nlls(), "failed": failed}
            out[model] = {"selected": choice, "variants": rows}
        self.data["sensitivity"] = out
        return lines

    def section_decoding(self) -> list[str]:
        d = self.study.decoding
        lines = [
            "## 5. Decoding sensitivity",
            "",
            f"Mean {d.split} BLEU over seeds {list(d.seeds)} (averaged checkpoint). The valid-best "
            "setting per model is used in section 2; test data never drives the choice.",
        ]
        out: dict[str, Any] = {}
        alphas = d.length_penalties
        for model in self.study.models:
            grid: dict[str, float | None] = {}
            for spec in grid_evals(self.study):
                values = [
                    float(self.states[n].evals[spec.tag]["bleu"])
                    for n in (main_run(self.study, model, s).name for s in d.seeds)
                    if n in self.states and spec.tag in self.states[n].evals
                ]
                grid[spec.tag] = _mean(values)
            choice = select_decoding(self.study, model, self.states)
            lines += [
                "",
                f"**{model}** (selected: {_decode_label(choice)})",
                "",
                "| Beam | " + " | ".join(f"α={a:g}" for a in alphas) + " |",
                "|---|" + "---|" * len(alphas),
            ]
            for beam in d.beams:
                cells = []
                for a in alphas:
                    tag = EvalSpec(
                        d.split, "averaged", beam, a if beam > 1 else self.study.default_alpha
                    ).tag
                    cells.append(_f(grid.get(tag)))
                    if beam == 1:
                        cells = [cells[0]] + ["″"] * (len(alphas) - 1)
                        break
                lines.append(
                    f"| {beam if beam > 1 else '1 (greedy)'} | " + " | ".join(cells) + " |"
                )
            out[model] = {
                "grid": grid,
                "selected": None if choice is None else [choice.beam, choice.alpha],
            }
        self.data["decoding"] = out
        return lines

    def section_free_ablations(self) -> list[str]:
        lines = [
            "## 6. Checkpoint averaging and beam search",
            "",
            "Paired on the same runs (all main seeds); Δ = first − second; "
            "Holm over the whole table.",
            "",
            "| Model | Comparison | Δ BLEU | 95% CI (Δ) | p | p (Holm) |",
            "|---|---|---|---|---|---|",
        ]
        beam = self.study.default_beam
        pairs = []
        for model in self.study.models:
            runs = self.main_system(model).runs
            averaged = System("averaged", runs, default_test(self.study, "averaged").tag)
            best = System("best", runs, default_test(self.study, "best").tag)
            greedy = System(
                "greedy", runs, EvalSpec("test", "best", 1, self.study.default_alpha).tag
            )
            pairs.append((model, f"averaged − best (beam {beam})", averaged, best))
            pairs.append((model, f"beam {beam} − greedy (best)", best, greedy))
        tests = [self.compare(a, b) for _, _, a, b in pairs]
        adjusted = holm([t["p_bootstrap"] if t else None for t in tests])
        out = []
        for (model, label, _, _), test, p_adj in zip(pairs, tests, adjusted, strict=True):
            if test is None:
                lines.append(f"| {model} | {label} | - | - | - | - |")
                continue
            lines.append(
                f"| {model} | {label} | {test['delta']:+.2f} | {_ci(test)} | "
                f"{format_p(test['p_bootstrap'], self.n)} | {format_p(p_adj, self.n)} |"
            )
            out.append({"model": model, "comparison": label, **test, "p_holm": p_adj})
        self.data["averaging_and_search"] = out
        return lines

    def section_ablations(self) -> list[str]:
        group = self.study.ablations
        lines = [
            "## 7. Ablations",
            "",
            f"Each variant (seeds {list(group.seeds)}) against its model's default "
            f"(seeds {list(self.study.main_seeds)}); test, averaged checkpoint, protocol decoding. "
            "Holm over all ablations of a model.",
        ]
        out = []
        for model in group.models:
            reference = self.main_system(model)
            systems = []
            for v in group.variants:
                names = [variant_run(self.study, model, "ablation", v, s).name for s in group.seeds]
                systems.append((v, System(v.name, self.runs(names), default_test(self.study).tag)))
            tests = [self.compare(s, reference) for _, s in systems]
            adjusted = holm([t["p_bootstrap"] if t else None for t in tests])
            ref_bleu = reference.values("bleu")
            lines += [
                "",
                f"**{model}**: default {_ms(ref_bleu)} BLEU, valid NLL {_ms(reference.nlls(), 3)}",
                "",
                "| Variant | Overrides | Params | Seeds | Valid NLL (ppl) | **BLEU** | ΔBLEU "
                "| 95% CI (Δ) | p | p (Holm) | p (seed perm.) |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for (v, system), test, p_adj in zip(systems, tests, adjusted, strict=True):
                bleu, nlls = system.values("bleu"), system.nlls()
                ppl = math.exp(_mean(nlls)) if nlls else None  # type: ignore[arg-type]
                params = system.scored[0].parameters if system.scored else None
                failed = sum(r.failed for r in system.runs)
                seeds = f"{len(bleu)}/{len(group.seeds)}" + (
                    f" ({failed} failed)" if failed else ""
                )
                if test is None:
                    stats = "- | - | - | - | -"
                else:
                    stats = (
                        f"{test['delta']:+.2f} | [{test['ci'][0]:+.2f}, {test['ci'][1]:+.2f}] | "
                        f"{format_p(test['p_bootstrap'], self.n)} | {format_p(p_adj, self.n)} | "
                        f"{test['p_permutation']:.3f}"
                    )
                lines.append(
                    f"| {v.name} | {' '.join(f'`{o}`' for o in v.overrides)} | "
                    f"{_f(params, ',') if params else '-'} | {seeds} | "
                    f"{_ms(nlls, 3)} ({_f(ppl)}) | "
                    f"**{_ms(bleu)}** | {stats} |"
                )
                out.append(
                    {
                        "model": model,
                        "variant": v.name,
                        "bleu": bleu,
                        "valid_nll": nlls,
                        "test": test,
                        "p_holm": p_adj,
                        "failed": failed,
                    }
                )
        self.data["ablations"] = out
        return lines

    def section_notes(self) -> list[str]:
        tag = default_test(self.study).tag
        usable = [
            s
            for s in self.states.values()
            if s.usable and tag in s.evals and s.best_valid_nll is not None
        ]
        lines = ["## 8. Statistical notes and checks", ""]
        if len(usable) >= 3:
            rho = spearman(
                [float(s.best_valid_nll or 0.0) for s in usable],
                [float(s.evals[tag]["bleu"]) for s in usable],
            )
            self.data["spearman_valid_nll_test_bleu"] = {"rho": rho, "runs": len(usable)}
            lines.append(
                f"- Validation NLL vs test BLEU over {len(usable)} runs: Spearman ρ = {rho:+.3f} "
                "(selection on validation NLL is meaningful only if this is strongly negative)."
            )
        seed_stds, sampling_stds = [], []
        for model in self.study.models:
            system = self.main_system(model)
            bleu = system.values("bleu")
            if len(bleu) > 1:
                seed_stds.append(mean_std(bleu)[1])
                sampling_stds += [
                    float(np.std(self.bootstrap.scores(h)[1], ddof=1)) for h in system.hypotheses()
                ]
        if seed_stds and sampling_stds:
            lines.append(
                "- Variance sources (main runs): seed-to-seed BLEU std "
                f"{sum(seed_stds) / len(seed_stds):.2f}; test-sampling std of a single run "
                f"{sum(sampling_stds) / len(sampling_stds):.2f}. "
                "Single-seed confidence intervals capture only the second."
            )
        fingerprints = {
            s.metadata.get("tokenizer_fingerprint") for s in self.states.values() if s.usable
        }
        data_configs = {
            json.dumps(load_yaml(s.path / "config.yaml")["data"], sort_keys=True)
            for s in self.states.values()
            if s.usable
        }
        lines += [
            f"- {_check(len(fingerprints) <= 1)}: one tokenizer fingerprint across all runs.",
            f"- {_check(len(data_configs) <= 1)}: one data configuration across all runs.",
            "- Declared differences: RNN experiments use `train.clip_norm=1.0` (ADR-021); "
            "variants differ only by their listed overrides; seeds by `train.seed`.",
            f"- Tests: two-level paired bootstrap ({self.n} samples; sentences × seeds), p floor "
            f"{format_p(1 / (self.n + 1), self.n)}; Holm-Bonferroni within each table; exact "
            "seed permutation test as a training-variance-only check (its smallest attainable p "
            "depends on the number of seeds).",
        ]
        return lines


def build_study_report(
    study: StudyConfig, states: Mapping[str, RunState], references: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    """Markdown report and a machine-readable dict (for figures and docs/results.md)."""
    analyzer = Analyzer(study, states, references)
    planned = {r.name for r in planned_runs(study)}
    known = set(all_run_names(study))
    completed = sum(1 for n in known if n in states and states[n].usable)
    failed = sorted(n for n in known if n in states and states[n].failed)
    pending = tasks(study, states)
    missing = sorted(n for n in planned if n not in states)
    header = [
        f"# Study `{study.name}`",
        "",
        f"Reference: `{study.reference}`. "
        "Protocol: ADR-021 (fairness), ADR-022 (study and statistics).",
        "",
        f"**Status:** {completed} runs completed, {len(failed)} failed, "
        f"{len(missing)} planned runs not started, {len(pending)} tasks pending"
        + (" — **partial results**." if pending else " — complete."),
    ]
    if failed:
        header.append(
            f"Failed runs (diverged or crashed; reported, not retried): {', '.join(failed)}."
        )
    sections = [
        header,
        analyzer.section_main(),
        analyzer.section_decoding_selected(),
        analyzer.section_tuned(),
        analyzer.section_sensitivity(),
        analyzer.section_decoding(),
        analyzer.section_free_ablations(),
        analyzer.section_ablations(),
        analyzer.section_notes(),
    ]
    analyzer.data["status"] = {
        "completed": completed,
        "failed": failed,
        "missing": missing,
        "pending_tasks": len(pending),
    }
    text = "\n\n".join("\n".join(s) for s in sections) + "\n"
    return text, analyzer.data


def write_study_report(
    study: StudyConfig, states: Mapping[str, RunState], references: Sequence[str], out: Path
) -> Path:
    text, data = build_study_report(study, states, references)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out
