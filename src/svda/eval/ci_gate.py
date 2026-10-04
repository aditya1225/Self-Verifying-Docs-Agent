"""CI gate (Ch. 19): fail the PR when quality regresses.

Two kinds of check:

* **Absolute floors** - invariants that must hold on any commit (an answer
  without citations is a bug, not a regression).
* **Relative tolerance** - each metric may not fall more than `tolerance`
  below the recorded baseline, so normal run-to-run noise doesn't block a PR
  but a real drop does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..schemas import EvalReport, ProfileMetrics

# Metric -> absolute minimum. Latency/cost use maximums and are listed separately.
FLOORS: dict[str, float] = {
    "faithfulness": 0.70,
    "citation_coverage": 0.95,
    "answer_accuracy": 0.50,
    "refusal_correctness": 0.80,
    "retrieval_recall": 0.50,
}

CEILINGS: dict[str, float] = {
    "p95_latency_ms": 60_000.0,
}

DEFAULT_TOLERANCE = 0.03  # 3 percentage points


@dataclass
class GateFinding:
    metric: str
    kind: str  # "floor" | "ceiling" | "regression"
    actual: float
    expected: float

    def describe(self) -> str:
        if self.kind == "floor":
            return f"{self.metric}: {self.actual:.3f} is below the required floor {self.expected:.3f}"
        if self.kind == "ceiling":
            return f"{self.metric}: {self.actual:.1f} exceeds the ceiling {self.expected:.1f}"
        return (
            f"{self.metric}: {self.actual:.3f} regressed from the baseline "
            f"{self.expected:.3f} by more than the tolerance"
        )


@dataclass
class GateResult:
    passed: bool
    findings: list[GateFinding]
    baseline_found: bool
    profile: str

    def report(self) -> str:
        if self.passed:
            head = f"CI gate PASSED on profile '{self.profile}'"
            if not self.baseline_found:
                head += " (no baseline recorded; floors only)"
            return head
        lines = [f"CI gate FAILED on profile '{self.profile}':"]
        lines += [f"  - {f.describe()}" for f in self.findings]
        return "\n".join(lines)


def load_baseline(path: Path | None = None) -> ProfileMetrics | None:
    path = path or get_settings().baseline_path
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProfileMetrics.model_validate(data["metrics"])


def check(
    report: EvalReport,
    *,
    profile: str = "full",
    baseline: ProfileMetrics | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    baseline_path: Path | None = None,
) -> GateResult:
    metrics = report.metrics_for(profile)
    if metrics is None:
        return GateResult(
            passed=False,
            findings=[GateFinding(profile, "floor", 0.0, 1.0)],
            baseline_found=False,
            profile=profile,
        )

    findings: list[GateFinding] = []

    for metric, floor in FLOORS.items():
        actual = float(getattr(metrics, metric))
        if actual + 1e-9 < floor:
            findings.append(GateFinding(metric, "floor", actual, floor))

    for metric, ceiling in CEILINGS.items():
        actual = float(getattr(metrics, metric))
        if actual > ceiling:
            findings.append(GateFinding(metric, "ceiling", actual, ceiling))

    if baseline is None:
        baseline = load_baseline(baseline_path)
    if baseline is not None:
        for metric in FLOORS:
            actual = float(getattr(metrics, metric))
            expected = float(getattr(baseline, metric))
            if actual + tolerance + 1e-9 < expected:
                findings.append(GateFinding(metric, "regression", actual, expected))

    return GateResult(
        passed=not findings,
        findings=findings,
        baseline_found=baseline is not None,
        profile=profile,
    )
