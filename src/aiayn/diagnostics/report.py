"""Check results and the machine-readable validation report."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = ["CheckResult", "Outcome", "Report", "Status"]

Status = Literal["pass", "warn", "fail", "skip"]


@dataclass(frozen=True)
class Outcome:
    """What a check function returns."""

    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    critical: bool
    message: str
    details: dict[str, Any]
    seconds: float


class Report:
    """Ordered collection of check results with JSON export and a console table."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []
        self.environment: dict[str, Any] = {}
        self._start = time.perf_counter()

    def run(self, name: str, fn: Callable[[], Outcome], *, critical: bool = True) -> CheckResult:
        """Run ``fn``; an exception becomes a ``fail`` with the traceback tail in details."""
        start = time.perf_counter()
        try:
            outcome = fn()
        except Exception as exc:  # a broken check must not crash the whole report
            tail = traceback.format_exc().strip().splitlines()[-6:]
            outcome = Outcome("fail", f"{type(exc).__name__}: {exc}", {"traceback": tail})
        result = CheckResult(
            name=name,
            status=outcome.status,
            critical=critical,
            message=outcome.message,
            details=outcome.details,
            seconds=round(time.perf_counter() - start, 3),
        )
        self.results.append(result)
        return result

    def skip(self, name: str, reason: str, *, critical: bool = True) -> CheckResult:
        return self.run(name, lambda: Outcome("skip", reason), critical=critical)

    def get(self, name: str) -> CheckResult | None:
        return next((r for r in self.results if r.name == name), None)

    def passed(self, name: str) -> bool:
        result = self.get(name)
        return result is not None and result.status in ("pass", "warn")

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.critical and r.status == "fail"]

    @property
    def total_seconds(self) -> float:
        return round(time.perf_counter() - self._start, 1)

    def exit_code(self) -> int:
        return 1 if self.critical_failures else 0

    def to_dict(self) -> dict[str, Any]:
        counts = {
            s: sum(r.status == s for r in self.results) for s in ("pass", "warn", "fail", "skip")
        }
        return {
            "ok": not self.critical_failures,
            "total_seconds": self.total_seconds,
            "counts": counts,
            "environment": self.environment,
            "checks": [asdict(r) for r in self.results],
        }

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return out

    def format_table(self) -> str:
        width = max((len(r.name) for r in self.results), default=10)
        lines = [f"{'check':<{width}}  status  crit   secs  message", "-" * (width + 60)]
        for r in self.results:
            crit = "yes" if r.critical else "no"
            lines.append(
                f"{r.name:<{width}}  {r.status:<6}  {crit:<4} {r.seconds:>6.1f}  {r.message}"
            )
        summary = self.to_dict()["counts"]
        verdict = "OK" if not self.critical_failures else "FAILED"
        lines.append("-" * (width + 60))
        lines.append(f"{verdict}: {summary} in {self.total_seconds}s")
        return "\n".join(lines)
