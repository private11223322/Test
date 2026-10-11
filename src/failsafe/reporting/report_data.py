"""Build a report payload from run results, findings, and metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from failsafe.evaluation.metrics import MetricsReport, RunResult
from failsafe.schemas import DetectorFinding, FindingStatus, Trace

STATUS_LABELS = {
    FindingStatus.VERIFIED: "Verified",
    FindingStatus.RULE_TRIGGERED: "Rule-triggered (needs context)",
    FindingStatus.SEMANTIC_SUSPECT: "Semantic suspicion (unverified)",
    FindingStatus.UNRESOLVED: "Unresolved (human review)",
    FindingStatus.DISMISSED: "Dismissed",
    FindingStatus.CONFIRMED_BY_HUMAN: "Confirmed by human",
    FindingStatus.REJECTED_BY_HUMAN: "Rejected by human",
}

STATUS_CLASSES = {
    FindingStatus.VERIFIED: "verified",
    FindingStatus.RULE_TRIGGERED: "rule",
    FindingStatus.SEMANTIC_SUSPECT: "semantic",
    FindingStatus.UNRESOLVED: "unresolved",
    FindingStatus.DISMISSED: "dismissed",
    FindingStatus.CONFIRMED_BY_HUMAN: "verified",
    FindingStatus.REJECTED_BY_HUMAN: "dismissed",
}

KNOWN_LIMITATIONS = [
    "Deterministic detectors use structural evidence only; semantically equivalent but "
    "structurally different behaviors may be missed or misread.",
    "LLM judge findings are unverified suspicions and never control the CI gate on their own.",
    "A fixed fault-injection seed reproduces the schedule, not live LLM behavior.",
    "Recovery is credited only when the oracle verifies the task outcome.",
]


def _finding_dict(f: DetectorFinding) -> dict[str, Any]:
    return {
        "id": f.id,
        "detector_id": f.detector_id,
        "category": f.category,
        "severity": f.severity.value,
        "status": f.status.value,
        "status_label": STATUS_LABELS.get(f.status, f.status.value),
        "status_class": STATUS_CLASSES.get(f.status, "rule"),
        "summary": f.summary,
        "description": f.description,
        "step_ids": f.step_ids,
        "evidence": [e.model_dump(mode="json") for e in f.evidence],
        "metrics": f.metrics,
    }


def _trace_dict(trace: Trace) -> dict[str, Any]:
    return {
        "run_id": trace.metadata.run_id,
        "agent_name": trace.metadata.agent_name,
        "agent_version": trace.metadata.agent_version,
        "task_id": trace.metadata.task_id,
        "task_name": trace.metadata.task_name,
        "status": trace.metadata.status.value,
        "started_at": trace.metadata.started_at.isoformat(),
        "ended_at": trace.metadata.ended_at.isoformat() if trace.metadata.ended_at else None,
        "seed": trace.metadata.seed,
        "final_response": trace.metadata.final_response,
        "error": trace.metadata.error,
        "source_format": trace.source_format,
        "steps": [
            {
                "id": s.id,
                "kind": s.kind.value,
                "name": s.name,
                "status": s.status.value,
                "parent_id": s.parent_id,
                "timestamp": s.timestamp.isoformat(),
                "error_type": s.error_type,
                "error_message": s.error_message,
                "tool_name": s.tool_call.name if s.tool_call else None,
                "tool_success": s.tool_result.success if s.tool_result else None,
            }
            for s in trace.ordered_steps()
        ],
    }


def build_report_payload(
    *,
    title: str,
    trace: Trace | None = None,
    findings: Sequence[DetectorFinding] = (),
    run_result: RunResult | None = None,
    metrics: MetricsReport | None = None,
    investigation: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    regression_tests: Sequence[dict[str, Any]] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the report payload shared by Markdown and HTML renderers."""
    categories: dict[str, list[dict[str, Any]]] = {}
    finding_dicts = [_finding_dict(f) for f in findings]
    for fd in finding_dicts:
        categories.setdefault(fd["status_label"], []).append(fd)

    payload: dict[str, Any] = {
        "title": title,
        "generated_by": "Failsafe",
        "trace": _trace_dict(trace) if trace is not None else None,
        "findings": finding_dicts,
        "findings_by_status": {k: v for k, v in categories.items()},
        "run_result": None,
        "metrics": metrics.to_dict() if metrics is not None else None,
        "investigation": investigation,
        "gate": gate,
        "regression_tests": list(regression_tests),
        "known_limitations": KNOWN_LIMITATIONS,
        "extra": extra or {},
    }

    if run_result is not None:
        rr = run_result
        payload["run_result"] = {
            "run_id": rr.run_id,
            "task_id": rr.task_id,
            "task_name": rr.task_name,
            "task_success": rr.task_success,
            "recovered": rr.recovered,
            "safe_handled": rr.safe_handled,
            "silent_failure": rr.silent_failure,
            "faults_scheduled": rr.faults_scheduled,
            "faults_injected": rr.faults_injected,
            "runtime_ms": rr.runtime_ms,
            "error": rr.error,
            "oracle": {
                "success": rr.oracle.success,
                "detail": rr.oracle.detail,
                "checks": [c.model_dump(mode="json") for c in rr.oracle.checks],
            },
            "findings": [_finding_dict(f) for f in rr.findings],
            "fault_injections": [fi.model_dump(mode="json") for fi in rr.trace.fault_injections],
        }
    return payload
