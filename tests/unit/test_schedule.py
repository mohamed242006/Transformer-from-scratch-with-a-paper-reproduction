"""Learning-rate schedules vs the closed forms of math.md §7."""

from __future__ import annotations

import pytest
import torch

from aiayn.training.config import ScheduleConfig
from aiayn.training.schedule import LRSchedule, inverse_sqrt_lr, noam_lr


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (1, 1.746928e-7),
        (1000, 1.746928e-4),
        (4000, 6.987712e-4),
        (10000, 4.419417e-4),
        (100000, 1.397542e-4),
    ],
)
def test_noam_matches_math_md_table(step: int, expected: float) -> None:
    assert noam_lr(step, 512, 4000) == pytest.approx(expected, rel=1e-6)


def test_noam_peaks_at_warmup_and_is_linear_then_inverse_sqrt() -> None:
    values = [noam_lr(s, 512, 100) for s in range(1, 1001)]
    assert max(range(1000), key=values.__getitem__) == 99  # step 100
    assert values[49] == pytest.approx(values[99] / 2)  # linear warmup
    assert values[399] == pytest.approx(values[99] / 2)  # 1/sqrt decay: step 400 = half


def test_inverse_sqrt_equals_noam_with_matching_peak() -> None:
    peak = (512 * 4000) ** -0.5
    for step in (1, 777, 4000, 9000, 123456):
        assert inverse_sqrt_lr(step, peak, 4000) == pytest.approx(noam_lr(step, 512, 4000))


@pytest.mark.parametrize(
    "fn", [lambda: noam_lr(0, 512, 4000), lambda: inverse_sqrt_lr(0, 1e-3, 10)]
)
def test_step_zero_is_rejected(fn: object) -> None:
    with pytest.raises(ValueError):
        fn()  # type: ignore[operator]


def test_schedule_object_sets_every_param_group() -> None:
    params = [torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.Adam([{"params": params[:1]}, {"params": params[1:]}], lr=0.0)
    schedule = LRSchedule(ScheduleConfig(kind="inverse_sqrt", warmup_steps=10, peak_lr=1e-3), 64)
    lr = schedule.apply(opt, 10)
    assert lr == pytest.approx(1e-3) and all(g["lr"] == lr for g in opt.param_groups)
    assert schedule.peak_lr == pytest.approx(1e-3)
    noam = LRSchedule(ScheduleConfig(kind="noam", warmup_steps=4000), 512)
    assert noam.peak_lr == pytest.approx(6.987712e-4, rel=1e-6)
    constant = LRSchedule(ScheduleConfig(kind="constant", peak_lr=3e-4), 512)
    assert constant.lr_at(1) == constant.lr_at(10**6) == 3e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "noam", "peak_lr": 1e-3},
        {"kind": "inverse_sqrt"},
        {"warmup_steps": 0},
        {"kind": "constant", "peak_lr": -1.0},
    ],
)
def test_invalid_schedule_configs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ScheduleConfig(**kwargs)  # type: ignore[arg-type]
