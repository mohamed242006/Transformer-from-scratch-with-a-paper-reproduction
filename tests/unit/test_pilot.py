"""Pilot report and go/no-go gates (ADR-023 §8) on synthetic run directories of the real
``week9_pilot`` study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.pilot_report import main as pilot_main

from aiayn.study import StudyEnv, collect_states, load_study
from aiayn.study.pilot import build_pilot_report, validation_report_summary, write_pilot_report
from aiayn.study.plan import default_valid, main_run, sensitivity_variants, variant_run

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "configs" / "study" / "week9_pilot.yaml"


@pytest.fixture(scope="module")
def study() -> Any:
    return load_study(STUDY)


def make_run(
    work: Path,
    name: str,
    valid_nll: list[float],
    *,
    bleu: float,
    tag: str,
    tokens: float = 20000.0,
    minutes: float = 30.0,
    skipped: int = 0,
) -> None:
    run = work / name
    (run / "eval").mkdir(parents=True)
    (run / "config.yaml").write_text("name: x\n", encoding="utf-8")
    meta = {
        "best_valid_nll": min(valid_nll),
        "sessions": [
            {
                "status": "completed",
                "start": "2026-11-20T00:00:00+00:00",
                "end": f"2026-11-20T00:{int(minutes):02d}:00+00:00",
                "start_update": 0,
                "end_update": 6000,
            }
        ],
    }
    (run / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = []
    for i, nll in enumerate(valid_nll, start=1):
        step = i * 1000
        for k in range(10):
            lines.append(
                {
                    "split": "train",
                    "step": step - 900 + k * 100,
                    "nll": nll + 0.2,
                    "tgt_tokens_per_s": tokens,
                    "max_mem_mib": 2000 + k,
                    "grad_norm": 1.0 + k / 10,
                    "skipped": skipped,
                }
            )
        lines.append({"split": "valid", "step": step, "nll": nll})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", "utf-8")
    (run / "eval" / f"{tag}.json").write_text(json.dumps({"bleu": bleu}), encoding="utf-8")


def populate(study: Any, work: Path, *, post_curve: list[float], pre_curve: list[float]) -> None:
    tag = default_valid(study).tag
    bleu = {  # transformer: default best (interior); GRU: 1.4e-3 (interior); Luong: 2.8e-3 (edge)
        ("transformer", "default"): 33.0,
        ("transformer", "lr0.0005"): 31.0,
        ("transformer", "lr0.001"): 32.5,
        ("gru_bahdanau", "default"): 28.0,
        ("gru_bahdanau", "lr0.0014"): 30.0,
        ("gru_bahdanau", "lr0.0028"): 29.0,
        ("bilstm_luong", "default"): 27.0,
        ("bilstm_luong", "lr0.0014"): 28.0,
        ("bilstm_luong", "lr0.0028"): 29.5,
    }
    for model in study.models:
        curve = post_curve if model == "transformer" else [6.0, 5.0, 4.5, 4.2, 4.0, 3.9]
        make_run(
            work, main_run(study, model, 1).name, curve, bleu=bleu[(model, "default")], tag=tag
        )
        for v in sensitivity_variants(study, model):
            name = variant_run(study, model, "sensitivity", v, 1).name
            make_run(work, name, [n + 0.05 for n in curve], bleu=bleu[(model, v.name)], tag=tag)
    pre_ln = study.ablations.variants[0]
    make_run(
        work,
        variant_run(study, "transformer", "ablation", pre_ln, 1).name,
        pre_curve,
        bleu=32.8,
        tag=tag,
    )


IMPROVING = [5.0, 4.0, 3.5, 3.2, 3.0, 2.9]


def test_go_decision_learning_rates_and_edges(study: Any, tmp_path: Path) -> None:
    populate(study, tmp_path, post_curve=IMPROVING, pre_curve=[n + 0.02 for n in IMPROVING])
    text, data = build_pilot_report(study, collect_states(study, StudyEnv(tmp_path)))
    gates = data["gates"]
    assert gates["G1"]["ok"] and gates["G2"]["ok"] and gates["post_ln_go"] is True
    assert "**GO**" in text and "`model.norm_position=post`" in text
    sel = data["selection"]
    assert (sel["transformer"]["peak_lr"], sel["gru_bahdanau"]["peak_lr"]) == ("0.0007", "0.0014")
    assert sel["bilstm_luong"]["peak_lr"] == "0.0028" and sel["bilstm_luong"]["edges"]
    assert gates["G3"]["ok"] is False and "beyond the grid edge" in text
    assert len(data["curves"]) == 6 and data["curves"][-1]["transformer:default"] == 2.9
    # 30 min per 6,000 updates -> 5 min/1k -> 2.5 h per 30k-update Phase 2 run.
    assert gates["G4"]["projected_hours"]["transformer"] == pytest.approx(2.5)
    assert gates["G4"]["ok"] is True


def test_no_go_when_post_ln_plateaus(study: Any, tmp_path: Path) -> None:
    plateau = [5.0, 4.0, 3.6, 3.5, 3.55, 3.6]  # best before the end: G1 fails
    populate(study, tmp_path, post_curve=plateau, pre_curve=IMPROVING)
    text, data = build_pilot_report(study, collect_states(study, StudyEnv(tmp_path)))
    assert data["gates"]["G1"]["ok"] is False and data["gates"]["G2"]["ok"] is False
    assert data["gates"]["post_ln_go"] is False
    assert "**NO-GO**" in text and "`model.norm_position=pre`" in text


def test_g2_margin_and_skipped_updates(study: Any, tmp_path: Path) -> None:
    populate(study, tmp_path, post_curve=IMPROVING, pre_curve=[n - 0.04 for n in IMPROVING])
    _, data = build_pilot_report(study, collect_states(study, StudyEnv(tmp_path)))
    assert data["gates"]["G2"]["ok"] is True  # Post worse by 0.04 nats: within +0.05


def test_pending_and_script_outputs(study: Any, tmp_path: Path) -> None:
    work = tmp_path / "runs"
    make_run(
        work,
        main_run(study, "transformer", 1).name,
        IMPROVING,
        bleu=33.0,
        tag=default_valid(study).tag,
    )
    _, data = build_pilot_report(study, collect_states(study, StudyEnv(work)))
    assert data["gates"]["post_ln_go"] is None and data["gates"]["G1"]["ok"] is True
    out = tmp_path / "PILOT.md"
    code = pilot_main(["--study", str(STUDY), "--work", str(work), "--out", str(out)])
    assert code == 3 and out.is_file() and (tmp_path / "pilot_curves.csv").is_file()
    assert (tmp_path / "PILOT.json").is_file()


def test_validation_report_summary(study: Any, tmp_path: Path) -> None:
    report = {
        "checks": [
            {
                "name": "model[transformer_iwslt].build",
                "status": "pass",
                "details": {"parameters": 36663296},
            },
            {
                "name": "model[transformer_iwslt].memory",
                "status": "pass",
                "details": {
                    "peak_allocated_mib": 2243.0,
                    "peak_reserved_mib": 2600.0,
                    "device_total_mib": 15095.0,
                },
            },
            {
                "name": "model[transformer_iwslt].throughput",
                "status": "pass",
                "details": {"target_tokens_per_s": 24669, "ms_per_step": 300.0},
            },
            {"name": "env.python", "status": "pass", "details": {}},
        ]
    }
    summary = validation_report_summary(report)
    assert summary == {
        "transformer_iwslt": {
            "parameters": 36663296,
            "peak_allocated_mib": 2243.0,
            "peak_reserved_mib": 2600.0,
            "device_total_mib": 15095.0,
            "memory_status": "pass",
            "target_tokens_per_s": 24669,
            "ms_per_step": 300.0,
        }
    }
    populate(study, tmp_path / "w", post_curve=IMPROVING, pre_curve=IMPROVING)
    path = write_pilot_report(
        study,
        collect_states(study, StudyEnv(tmp_path / "w")),
        tmp_path / "P.md",
        validation_report=report,
    )
    assert "Kaggle validation" in path.read_text("utf-8")
