"""Unit tests for the deterministic CI gate and regression-test generation."""

from __future__ import annotations

from failsafe.agents.regression import (
    RegressionTestValidator,
    generate_from_findings,
    generate_regression_test,
    regression_reproduces_issue,
    task_from_regression_test,
    validate_test,
)
from failsafe.agents.sample import get_agent
from failsafe.ci import evaluate_gate
from failsafe.config import CIGateConfig
from failsafe.evaluation import execute_task, get_builtin_tasks
from failsafe.schemas import (
    DetectorFinding,
    Evidence,
    FindingStatus,
    RegressionTest,
    RegressionTestState,
    Severity,
    TaskSpec,
)


def _finding(status=FindingStatus.RULE_TRIGGERED, severity=Severity.HIGH, category="repeated_actions"):
    return DetectorFinding(
        run_id="r", detector_id=category, category=category, summary="bad loop",
        severity=severity, status=status,
        evidence=[Evidence(kind="k", step_ids=["s1"])],
    )


# ---- gate ---------------------------------------------------------------

def test_gate_passes_clean():
    d = evaluate_gate([])
    assert d.passed
    assert d.mode == "strict"


def test_gate_fails_on_configured_finding():
    d = evaluate_gate([_finding()])
    assert not d.passed
    assert d.blocking_findings
    assert any("bad loop" in r for r in d.reasons)


def test_gate_ignores_semantic_suspicions():
    d = evaluate_gate([_finding(status=FindingStatus.SEMANTIC_SUSPECT, severity=Severity.CRITICAL)])
    assert d.passed  # LLM judge can never block the build on its own


def test_gate_ignores_unresolved():
    d = evaluate_gate([_finding(status=FindingStatus.UNRESOLVED, severity=Severity.CRITICAL)])
    assert d.passed


def test_gate_respects_category_filter():
    cfg = CIGateConfig(fail_on_categories=["ignored_tool_errors"])
    d = evaluate_gate([_finding(category="repeated_actions")], cfg)
    assert d.passed
    d2 = evaluate_gate([_finding(category="ignored_tool_errors")], cfg)
    assert not d2.passed


def test_gate_respects_severity_threshold():
    cfg = CIGateConfig(fail_on_min_severity="critical")
    d = evaluate_gate([_finding(severity=Severity.HIGH)], cfg)
    assert d.passed
    d2 = evaluate_gate([_finding(severity=Severity.CRITICAL)], cfg)
    assert not d2.passed


def test_gate_report_only_mode():
    cfg = CIGateConfig(mode="report-only")
    d = evaluate_gate([_finding()], cfg)
    assert d.passed
    assert d.report_only
    assert d.reasons  # still reports


def test_gate_human_rejection_is_not_blocking():
    d = evaluate_gate([_finding(status=FindingStatus.REJECTED_BY_HUMAN)])
    assert d.passed
    d2 = evaluate_gate([_finding(status=FindingStatus.CONFIRMED_BY_HUMAN)])
    assert not d2.passed


def test_gate_fails_on_oracle_failure():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("buggy"), task)
    d = evaluate_gate([], run_results=[res])
    assert not d.passed
    assert any("oracle" in r for r in d.reasons)


def test_gate_fails_on_silent_failure():
    task = next(t for t in get_builtin_tasks() if t.id == "task-update-record")
    res = execute_task(get_agent("buggy"), task)
    d = evaluate_gate([], run_results=[res])
    assert not d.passed
    assert any("silent" in r for r in d.reasons)


def test_gate_passes_good_agent_runs():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("good"), task)
    d = evaluate_gate(res.findings, run_results=[res])
    assert d.passed


# ---- regression tests ---------------------------------------------------

def test_generate_regression_test_from_finding():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("looping"), task)
    assert res.findings
    f = next(x for x in res.findings if x.detector_id == "repeated_actions")
    test = generate_regression_test(res.trace, f)
    assert test.state == RegressionTestState.PROPOSED
    assert test.scenario
    assert test.reproduction
    assert test.original_finding_ids == [f.id]
    assert test.expected_behavior


def test_validate_regression_test():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("looping"), task)
    f = next(x for x in res.findings if x.detector_id == "repeated_actions")
    test = generate_regression_test(res.trace, f)
    validated, errors = validate_test(test)
    assert errors == []
    assert validated.state == RegressionTestState.VALIDATED


def test_invalid_regression_test_rejected():
    bad = RegressionTest(
        scenario="short",
        task=TaskSpec(name=""),
        oracle={"checks": [{"check": "nope"}]},
        expected_behavior="",
        reproduction="",
    )
    errors = RegressionTestValidator().validate(bad)
    assert len(errors) >= 4


def test_generate_from_findings_skips_suspicions():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("buggy"), task)
    # mark all findings as semantic suspicions
    for f in res.findings:
        f.status = FindingStatus.SEMANTIC_SUSPECT
    tests = generate_from_findings(res.trace, res.findings)
    assert tests == []


def test_generated_test_becomes_executable_task():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("looping"), task)
    f = next(x for x in res.findings if x.detector_id == "repeated_actions")
    test = generate_regression_test(res.trace, f)
    validated, _ = validate_test(test)
    exec_task = task_from_regression_test(validated)
    assert exec_task.oracle_checks
    res2 = execute_task(get_agent("looping"), exec_task)
    assert res2.run_id


def test_regression_reproduces_issue_semantics():
    assert regression_reproduces_issue(None, result_oracle_success=False) is True
    assert regression_reproduces_issue(None, result_oracle_success=True, detector_refired=True) is True
    assert regression_reproduces_issue(None, result_oracle_success=True, detector_refired=False) is False


def test_fault_schedule_in_generated_test():
    task = next(t for t in get_builtin_tasks() if t.id == "task-recoverable-fault")
    res = execute_task(get_agent("flaky"), task)
    f = next(x for x in res.findings if x.detector_id == "ignored_tool_errors")
    test = generate_regression_test(res.trace, f)
    validated, errors = validate_test(test)
    assert errors == []
    for fs in validated.fault_schedule:
        assert fs["kind"] in ("timeout", "error", "empty_response", "malformed_response",
                              "stale_response", "rate_limit", "partial_result", "schema_drift")
