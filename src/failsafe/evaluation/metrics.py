"""Reliability and recovery metrics.

Denominators are defined explicitly for every metric and exported alongside
the values.  Successful recovery is counted only when the task outcome is
verified by the oracle.  An explicit correct failure report is safe-handling,
not recovery.  A failed fault injection (the fault never reached the agent)
is counted as an injection failure, never as a recovery.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from failsafe.schemas import (
    DetectorFinding,
    OracleOutcome,
    Trace,
)


@dataclass
class RunResult:
    """The full result of executing one agent on one task."""

    run_id: str
    task_id: str
    task_name: str
    trace: Trace
    oracle: OracleOutcome
    findings: list[DetectorFinding] = field(default_factory=list)
    faults_scheduled: int = 0
    faults_injected: int = 0
    runtime_ms: float = 0.0
    llm_usage: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    agent_reported_failure: bool = False
    error: str | None = None

    @property
    def task_success(self) -> bool:
        return self.oracle.success

    @property
    def recovered(self) -> bool:
        """Successful recovery: a fault was actually injected AND the oracle
        verified the outcome AND the agent did not merely report failure.
        Injection failures (scheduled but never triggered) never count."""
        return self.faults_injected > 0 and self.oracle.success and not self.agent_reported_failure

    @property
    def safe_handled(self) -> bool:
        """Verified safe handling: the agent explicitly reported failure and the
        oracle verified that reporting failure was the correct outcome.
        Distinct from successful recovery."""
        return self.agent_reported_failure and self.oracle.success

    @property
    def silent_failure(self) -> bool:
        """Verified silent failure: oracle failed but the agent claimed success."""
        from failsafe.detectors.unsupported_success import claims_success

        claimed = self.trace.metadata.status.value == "success"
        text = self.trace.metadata.final_response or ""
        if text:
            claimed = claimed and claims_success(text)
        return (not self.oracle.success) and claimed and not self.agent_reported_failure


@dataclass
class MetricValue:
    name: str
    value: float | None
    numerator: int | None = None
    denominator: int | None = None
    denominator_definition: str = ""
    detail: str = ""


@dataclass
class MetricsReport:
    values: list[MetricValue] = field(default_factory=list)
    confusion_matrices: dict[str, dict[str, int]] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": [
                {
                    "name": v.name,
                    "value": v.value,
                    "numerator": v.numerator,
                    "denominator": v.denominator,
                    "denominator_definition": v.denominator_definition,
                    "detail": v.detail,
                }
                for v in self.values
            ],
            "confusion_matrices": self.confusion_matrices,
            "sample_counts": self.sample_counts,
        }

    def get(self, name: str) -> MetricValue | None:
        for v in self.values:
            if v.name == name:
                return v
        return None


def _rate(numer: int, denom: int) -> float | None:
    return (numer / denom) if denom else None


def compute_run_metrics(results: Sequence[RunResult]) -> MetricsReport:
    """Compute aggregate reliability metrics over a sequence of run results."""
    report = MetricsReport()
    n = len(results)
    report.sample_counts["runs"] = n
    if n == 0:
        return report

    successes = [r for r in results if r.task_success]
    fault_runs = [r for r in results if r.faults_scheduled > 0]
    injected_runs = [r for r in results if r.faults_injected > 0]
    injection_failures = [r for r in fault_runs if r.faults_injected == 0]
    recovered = [r for r in results if r.recovered]
    safe = [r for r in results if r.safe_handled]
    silent = [r for r in results if r.silent_failure]

    report.values.append(MetricValue(
        name="task_success_rate",
        value=_rate(len(successes), n),
        numerator=len(successes), denominator=n,
        denominator_definition="all runs in the evaluation",
    ))
    report.values.append(MetricValue(
        name="fault_recovery_rate",
        value=_rate(len(recovered), len(injected_runs)),
        numerator=len(recovered), denominator=len(injected_runs),
        denominator_definition="runs where at least one fault was actually injected (injection failures excluded)",
    ))
    report.values.append(MetricValue(
        name="safe_handling_rate",
        value=_rate(len(safe), n),
        numerator=len(safe), denominator=n,
        denominator_definition="all runs; safe handling = explicit failure report where the oracle verified failure-reporting was correct",
    ))
    report.values.append(MetricValue(
        name="verified_silent_failure_rate",
        value=_rate(len(silent), n),
        numerator=len(silent), denominator=n,
        denominator_definition="all runs; silent failure = oracle failure while agent claimed success",
    ))
    report.values.append(MetricValue(
        name="injection_failure_rate",
        value=_rate(len(injection_failures), len(fault_runs)),
        numerator=len(injection_failures), denominator=len(fault_runs),
        denominator_definition="runs with scheduled faults; injection failure = fault scheduled but never triggered",
    ))

    runtime = [r.runtime_ms for r in results]
    report.values.append(MetricValue(
        name="mean_runtime_ms",
        value=(sum(runtime) / len(runtime)) if runtime else None,
        denominator=n,
        denominator_definition="all runs",
    ))

    costs = [r.estimated_cost_usd for r in results if r.estimated_cost_usd is not None]
    report.values.append(MetricValue(
        name="total_estimated_cost_usd",
        value=sum(costs) if costs else None,
        denominator=len(costs),
        denominator_definition="runs with reported LLM cost (live LLM runs only)",
        detail="cost is provider-reported or estimated; empty when no LLM used",
    ))

    llm_calls = sum(int(r.llm_usage.get("calls", 0)) for r in results if r.llm_usage)
    report.values.append(MetricValue(
        name="llm_calls_total",
        value=float(llm_calls) if any(r.llm_usage for r in results) else None,
        denominator=n,
        denominator_definition="all runs",
    ))

    return report


def confusion_matrix(
    expected: Iterable[bool],
    predicted: Iterable[bool],
    labels: tuple[str, str] = ("negative", "positive"),
) -> dict[str, int]:
    """Confusion matrix with explicit labels: tn, fp, fn, tp."""
    tn = fp = fn = tp = 0
    for e, p in zip(expected, predicted, strict=False):
        if e and p:
            tp += 1
        elif e and not p:
            fn += 1
        elif not e and p:
            fp += 1
        else:
            tn += 1
    return {
        f"tn_{labels[0]}": tn,
        f"fp_{labels[1]}": fp,
        f"fn_{labels[1]}": fn,
        f"tp_{labels[1]}": tp,
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def detector_metrics(
    findings_by_run: dict[str, list[DetectorFinding]],
    expected_by_run: dict[str, list[dict[str, Any]]],
    match_category: bool = True,
) -> MetricsReport:
    """Detector precision/recall/F1 against expected findings.

    An expected finding matches a predicted finding when categories are equal
    (or match_category is False) and they share at least one evidence step ID,
    or, when expected carries ``"summary_contains"``, when the predicted
    summary contains that substring.
    """
    report = MetricsReport()
    tp = fp = fn = 0
    per_category: dict[str, dict[str, int]] = {}

    for run_id, predicted in findings_by_run.items():
        expected = expected_by_run.get(run_id, [])
        matched_pred: set[int] = set()
        matched_exp: set[int] = set()
        for ei, exp in enumerate(expected):
            for pi, pred in enumerate(predicted):
                if pi in matched_pred:
                    continue
                if match_category and exp.get("category") and exp["category"] != pred.category:
                    continue
                ok = False
                if exp.get("summary_contains") and exp["summary_contains"].lower() in pred.summary.lower():
                    ok = True
                exp_steps = set(exp.get("step_ids", []))
                if exp_steps and exp_steps & set(pred.step_ids):
                    ok = True
                if not exp.get("step_ids") and not exp.get("summary_contains") and exp.get("category") == pred.category:
                    ok = True
                if ok:
                    matched_pred.add(pi)
                    matched_exp.add(ei)
                    cat = pred.category
                    per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
                    per_category[cat]["tp"] += 1
                    break
            else:
                cat = exp.get("category", "unknown")
                per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
                per_category[cat]["fn"] += 1
                fn += 1
                continue
            tp += 1
        for pi, pred in enumerate(predicted):
            if pi not in matched_pred:
                fp += 1
                cat = pred.category
                per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
                per_category[cat]["fp"] += 1

    report.confusion_matrices = per_category
    prf = precision_recall_f1(tp, fp, fn)
    report.values.append(MetricValue(name="detector_precision", value=prf["precision"], numerator=tp, denominator=tp + fp, denominator_definition="true positive + false positive findings"))
    report.values.append(MetricValue(name="detector_recall", value=prf["recall"], numerator=tp, denominator=tp + fn, denominator_definition="true positive + false negative expected findings"))
    report.values.append(MetricValue(name="detector_f1", value=prf["f1"], numerator=None, denominator=None, denominator_definition="harmonic mean of precision and recall"))
    report.values.append(MetricValue(name="false_positive_rate_raw", value=_rate(fp, fp + tp), numerator=fp, denominator=fp + tp, denominator_definition="predicted findings that were not expected"))
    report.sample_counts = {"expected": sum(len(v) for v in expected_by_run.values()), "predicted": sum(len(v) for v in findings_by_run.values()), "tp": tp, "fp": fp, "fn": fn}
    return report


def replay_stability(results_a: Sequence[RunResult], results_b: Sequence[RunResult]) -> MetricValue:
    """Fraction of runs whose oracle verdict is identical across two replays."""
    by_id_a = {r.run_id: r for r in results_a}
    matches = total = 0
    for rb in results_b:
        ra = by_id_a.get(rb.run_id)
        if ra is None:
            continue
        total += 1
        if ra.oracle.success == rb.oracle.success:
            matches += 1
    return MetricValue(
        name="replay_stability",
        value=_rate(matches, total),
        numerator=matches, denominator=total,
        denominator_definition="runs executed in both replay passes",
    )


def live_run_variability(success_flags: Sequence[bool]) -> MetricValue:
    """Report the spread of live-run outcomes (deterministic replays should be
    1.0 stability; live LLM runs vary)."""
    n = len(success_flags)
    if n == 0:
        return MetricValue(name="live_run_variability", value=None, denominator=0, denominator_definition="live runs")
    successes = sum(1 for s in success_flags if s)
    return MetricValue(
        name="live_run_variability",
        value=_rate(successes, n),
        numerator=successes, denominator=n,
        denominator_definition="live runs (success fraction; variability = spread across repeats)",
        detail="success fraction across repeated live runs of the same task",
    )
