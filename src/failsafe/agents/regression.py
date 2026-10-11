"""Regression-test generation and validation.

Candidate tests are generated from observed failures.  Every generated test
is validated before acceptance (schema + oracle definition).  States:
``proposed -> validated -> failing|passing`` (or ``rejected``).  Generation
never executes anything; execution happens only through the controlled
runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from failsafe.evaluation.oracles import checker_names
from failsafe.evaluation.tasks import OracleTask
from failsafe.schemas import (
    DetectorFinding,
    FaultKind,
    FindingStatus,
    RegressionTest,
    RegressionTestState,
    TaskSpec,
    Trace,
)

VALID_OUTCOME_TYPES = {
    "tool_called", "tool_called_with", "tool_success", "response_contains",
    "step_count_min",
}

_CATEGORY_TEMPLATES: dict[str, dict[str, Any]] = {
    "repeated_actions": {
        "scenario": "Agent must not loop on identical tool calls.",
        "expected_behavior": "The agent completes the task without repeating an identical tool call "
                             "beyond the configured threshold.",
        "fault_schedule": [],
    },
    "excessive_retries": {
        "scenario": "Agent must respect the retry budget on persistent failures.",
        "expected_behavior": "The agent stops retrying after the budget and reports the failure.",
        "fault_schedule": [{"kind": "error", "tool": "*", "occurrence": 1, "params": {"message": "persistent failure"}}],
    },
    "ignored_tool_errors": {
        "scenario": "Agent must handle tool errors explicitly.",
        "expected_behavior": "After a tool error the agent retries, falls back, or reports the failure; "
                             "it never continues as if nothing happened.",
        "fault_schedule": [{"kind": "error", "tool": "*", "occurrence": 1}],
    },
    "premature_termination": {
        "scenario": "Agent must complete all required task outcomes before finishing.",
        "expected_behavior": "Every required outcome in the task spec has structural evidence in the trace.",
        "fault_schedule": [],
    },
    "unsupported_success": {
        "scenario": "Agent must not claim success without evidence.",
        "expected_behavior": "The final response claims success only when the oracle verifies the outcome.",
        "fault_schedule": [],
    },
    "lost_context": {
        "scenario": "Agent must not lose or repeat context mid-task.",
        "expected_behavior": "No identical request is issued repeatedly; context persists across steps.",
        "fault_schedule": [],
    },
}


def _derive_checks(trace: Trace, finding: DetectorFinding, tools_used: list[str]) -> list[dict[str, Any]]:
    """Derive oracle checks that fail while the defect is present.

    Priority:
    1. Category-specific checks (e.g. anti-loop bounds for repeated_actions).
    2. The original task's required outcomes (converted to trace-scoped checks).
    3. A weak fallback only when nothing else is available.
    """
    checks: list[dict[str, Any]] = []
    if finding.category == "repeated_actions" and tools_used:
        count = int(finding.metrics.get("count", 2))
        checks.append({"check": "trace_tool_max_calls", "tool": tools_used[0], "max": max(1, count - 1)})
    if finding.category in ("premature_termination", "unsupported_success") and trace.task:
        for outcome in trace.task.required_outcomes:
            kind = outcome.get("type")
            if kind == "tool_called":
                checks.append({"check": "trace_tool_called", "tool": outcome.get("tool")})
            elif kind == "tool_success":
                checks.append({"check": "trace_tool_succeeded", "tool": outcome.get("tool")})
            elif kind == "response_contains":
                checks.append({"check": "response_contains", "text": outcome.get("text")})
    if finding.category == "ignored_tool_errors":
        # expected behavior: the tool error is handled (the detector stays quiet)
        checks.append({"check": "detector_quiet", "detector_id": "ignored_tool_errors"})
    if finding.category == "repeated_actions":
        checks.append({"check": "detector_quiet", "detector_id": "repeated_actions"})
    if not checks and trace.task and trace.task.required_outcomes:
        for outcome in trace.task.required_outcomes:
            kind = outcome.get("type")
            if kind == "tool_called":
                checks.append({"check": "trace_tool_called", "tool": outcome.get("tool")})
            elif kind == "tool_success":
                checks.append({"check": "trace_tool_succeeded", "tool": outcome.get("tool")})
            elif kind == "response_contains":
                checks.append({"check": "response_contains", "text": outcome.get("text")})
    if not checks:
        checks = [{"check": "final_status", "status": "success"}]
    return checks


class RegressionTestValidator:
    """Validates regression tests before acceptance."""

    def validate(self, test: RegressionTest) -> list[str]:
        errors: list[str] = []
        if not test.scenario or len(test.scenario.strip()) < 10:
            errors.append("scenario must be a descriptive sentence (>= 10 chars).")
        if not test.task.name:
            errors.append("task.name is required.")
        if not test.oracle:
            errors.append("oracle must define at least one check.")
        else:
            checks = test.oracle.get("checks", [])
            if not checks:
                errors.append("oracle.checks must be non-empty.")
            for c in checks:
                name = c.get("check")
                if name not in checker_names():
                    errors.append(f"unknown oracle checker '{name}'; known: {checker_names()}")
        for outcome in test.task.required_outcomes:
            if outcome.get("type") not in VALID_OUTCOME_TYPES:
                errors.append(f"unknown required_outcome type: {outcome.get('type')}")
        for fs in test.fault_schedule:
            try:
                FaultKind(fs["kind"])
            except Exception:
                errors.append(f"unknown fault kind in fault_schedule: {fs.get('kind')}")
            if "occurrence" in fs and not isinstance(fs["occurrence"], int):
                errors.append("fault_schedule occurrence must be int.")
        if not test.expected_behavior:
            errors.append("expected_behavior is required.")
        if not test.reproduction:
            errors.append("reproduction instructions are required.")
        return errors


def generate_regression_test(
    trace: Trace,
    finding: DetectorFinding,
    oracle_checks: list[dict[str, Any]] | None = None,
) -> RegressionTest:
    """Generate a candidate regression test from an observed failure."""
    template = _CATEGORY_TEMPLATES.get(finding.category, {})
    task = trace.task or TaskSpec(name=f"regression-{finding.category}", description="generated")

    # Build a fault schedule from the finding evidence when the failure involved tools.
    fault_schedule: list[dict[str, Any]] = list(template.get("fault_schedule", []))
    tools_used = [tc.name for tc in trace.tool_calls()]
    if fault_schedule:
        for fs in fault_schedule:
            if fs.get("tool") == "*" and tools_used:
                fs["tool"] = tools_used[0]

    # Precondition: the original failing steps exist.
    preconditions = [
        f"Original run {trace.metadata.run_id} exhibits finding {finding.id} ({finding.category}).",
    ]
    for sid in finding.step_ids:
        preconditions.append(f"Original trace contains failing step {sid}.")

    checks = oracle_checks or _derive_checks(trace, finding, tools_used)

    task_model = TaskSpec(
        # preserve the original task id so the same scenario is re-executed
        # (agents dispatch on the task id; the scenario name stays unique)
        id=task.id,
        name=f"regression-{finding.category}-{finding.id[-6:]}",
        description=template.get("scenario", f"Regression for {finding.category}"),
        instruction=task.instruction or task.description or template.get("scenario", "Regression task"),
        required_outcomes=task.required_outcomes,
        tool_policy=task.tool_policy,
        # preserve world_kind etc. so the exact world can be reconstructed
        metadata=dict(task.metadata),
    )

    reproduction = (
        f"1. Recreate the world for task '{task.name or task.id}'.\n"
        f"2. Apply fault schedule {fault_schedule or '[]'}.\n"
        f"3. Run the agent under test.\n"
        f"4. Run oracle checks {checks}.\n"
        f"5. The test FAILS while the defect is present (oracle fails or detector "
        f"'{finding.detector_id}' re-fires), and PASSES once the defect is corrected."
    )

    test = RegressionTest(
        state=RegressionTestState.PROPOSED,
        scenario=template.get("scenario", f"Regression: {finding.summary}"),
        preconditions=preconditions,
        task=task_model,
        tools_required=list(dict.fromkeys(tools_used)),
        fault_schedule=fault_schedule,
        expected_behavior=template.get(
            "expected_behavior",
            "The defect that produced the original finding is not reproduced.",
        ),
        oracle={"checks": checks},
        reproduction=reproduction,
        original_finding_ids=[finding.id],
        original_run_id=trace.metadata.run_id,
        source="generated",
    )
    return test


def validate_test(test: RegressionTest) -> tuple[RegressionTest, list[str]]:
    """Validate a proposed test; on success move it to VALIDATED."""
    errors = RegressionTestValidator().validate(test)
    if not errors:
        test.state = RegressionTestState.VALIDATED
    return test, errors


def generate_from_findings(
    trace: Trace,
    findings: Sequence[DetectorFinding],
    oracle_checks: list[dict[str, Any]] | None = None,
    *,
    statuses: set[FindingStatus] | None = None,
) -> list[RegressionTest]:
    """Generate candidate tests for findings (default: verified/rule-triggered only)."""
    allowed = statuses or {FindingStatus.VERIFIED, FindingStatus.RULE_TRIGGERED, FindingStatus.CONFIRMED_BY_HUMAN}
    tests: list[RegressionTest] = []
    for f in findings:
        if f.status not in allowed:
            continue  # never generate tests from unverified suspicions alone
        test = generate_regression_test(trace, f, oracle_checks)
        validated, errors = validate_test(test)
        if errors:
            validated.state = RegressionTestState.REJECTED
            validated.last_result = {"validation_errors": errors}
        tests.append(validated)
    return tests


def task_from_regression_test(test: RegressionTest, world_factory: Any = None) -> OracleTask:
    """Convert a validated regression test into an executable OracleTask.

    The world is reconstructed from ``test.task.metadata["world_kind"]`` when
    no explicit factory is given, so generated tests run against the exact
    environment of the original failure.
    """
    from failsafe.evaluation.tasks import resolve_world_factory

    factory = world_factory or resolve_world_factory(test.task.metadata.get("world_kind"))
    return OracleTask(
        spec=test.task,
        world_factory=factory,
        oracle_checks=list(test.oracle.get("checks", [])),
        tools_required=list(test.tools_required),
        fault_schedule=list(test.fault_schedule),
        expected_behavior=test.expected_behavior,
        category="regression",
    )


def regression_reproduces_issue(
    test: RegressionTest,
    result_oracle_success: bool,
    detector_refired: bool = False,
) -> bool:
    """A generated test reproduces the original issue when the run fails the
    oracle (or the original detector fires again) — i.e. the test catches the
    defect while it is still present."""
    return (not result_oracle_success) or detector_refired
