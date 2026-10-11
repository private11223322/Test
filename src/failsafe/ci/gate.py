"""Deterministic CI gate.

The gate is driven exclusively by configured, deterministic criteria:
verified findings, oracle outcomes, and silent failures.  LLM judge output
(semantic suspicions) is reported but can never by itself block or approve a
build.  Two modes:

- ``strict``: fail the build on configured critical findings / oracle failures.
- ``report-only``: never fail; produce the report for exploratory evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from failsafe.config import CIGateConfig
from failsafe.evaluation.metrics import RunResult
from failsafe.schemas import DetectorFinding, FindingStatus, Severity

SEVERITY_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]

# Findings with these statuses are allowed to block a build (deterministic).
BLOCKING_STATUSES = {
    FindingStatus.VERIFIED,
    FindingStatus.RULE_TRIGGERED,
    FindingStatus.CONFIRMED_BY_HUMAN,
}

# Findings with these statuses can never block a build on their own.
NON_BLOCKING_STATUSES = {
    FindingStatus.SEMANTIC_SUSPECT,
    FindingStatus.UNRESOLVED,
    FindingStatus.DISMISSED,
    FindingStatus.REJECTED_BY_HUMAN,
}


@dataclass
class GateDecision:
    passed: bool
    mode: str
    reasons: list[str] = field(default_factory=list)
    blocking_findings: list[str] = field(default_factory=list)
    report_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mode": self.mode,
            "reasons": self.reasons,
            "blocking_findings": self.blocking_findings,
            "report_only": self.report_only,
        }


def _severity_at_least(sev: Severity, minimum: Severity) -> bool:
    return SEVERITY_ORDER.index(sev) >= SEVERITY_ORDER.index(minimum)


def evaluate_gate(
    findings: Sequence[DetectorFinding],
    config: CIGateConfig | None = None,
    run_results: Sequence[RunResult] | None = None,
) -> GateDecision:
    cfg = config or CIGateConfig()
    report_only = cfg.mode.lower() in ("report-only", "report_only", "report")
    reasons: list[str] = []
    blocking: list[str] = []

    min_sev = Severity(cfg.fail_on_min_severity)
    fail_categories = set(cfg.fail_on_categories)

    for f in findings:
        if f.status in NON_BLOCKING_STATUSES:
            # Explicitly never blocking; note for the report only.
            continue
        if f.status not in BLOCKING_STATUSES:
            continue
        if f.status == FindingStatus.CONFIRMED_BY_HUMAN:
            # A human-confirmed finding is authoritative: it bypasses the
            # category and severity filters and always blocks in strict mode.
            category_ok = severity_ok = True
        else:
            category_ok = (not fail_categories) or (f.category in fail_categories)
            severity_ok = _severity_at_least(f.severity, min_sev)
        if category_ok and severity_ok:
            blocking.append(f.id)
            reasons.append(
                f"[{f.category}] {f.summary} (severity={f.severity.value}, status={f.status.value}, "
                f"steps={','.join(f.step_ids) or 'n/a'})"
            )

    if run_results:
        for rr in run_results:
            if cfg.fail_on_oracle_failure and not rr.oracle.success:
                blocking.append(rr.oracle.id)
                reasons.append(
                    f"[oracle] Task '{rr.task_name or rr.task_id}' failed oracle verification "
                    f"(run={rr.run_id}): {rr.oracle.detail}"
                )
            if cfg.fail_on_silent_failure and rr.silent_failure:
                blocking.append(f"silent-{rr.run_id}")
                reasons.append(
                    f"[silent_failure] Run {rr.run_id}: agent claimed success but the oracle verified failure."
                )

    if report_only:
        return GateDecision(
            passed=True,
            mode="report-only",
            reasons=reasons or ["No blocking findings (report-only mode never fails the build)."],
            blocking_findings=blocking,
            report_only=True,
        )

    return GateDecision(
        passed=not blocking,
        mode=cfg.mode,
        reasons=reasons if blocking else ["All configured gate criteria satisfied."],
        blocking_findings=blocking,
        report_only=False,
    )
