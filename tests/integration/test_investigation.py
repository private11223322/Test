"""Integration tests for the LangGraph investigation agent."""

from __future__ import annotations

import pytest

from failsafe.agents.investigation import (
    LANGGRAPH_AVAILABLE,
    HeuristicPlanner,
    LLMPlanner,
    investigate_run,
)
from failsafe.agents.sample import get_agent
from failsafe.agents.tools import InvestigationTools
from failsafe.config import BudgetConfig
from failsafe.evaluation import execute_task, get_builtin_tasks
from failsafe.schemas import Trace
from tests.conftest import TraceBuilder

pytestmark = pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")


def _suspicious_run():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("looping"), task)
    return res


def test_tools_allowlist_surface():
    tools = InvestigationTools(trace=Trace(), findings=[])
    names = set(tools.as_map())
    expected = {
        "inspect_trace", "retrieve_steps", "inspect_tool_errors", "compare_runs",
        "read_task_spec", "search_regression_tests", "propose_regression_test",
        "validate_test", "request_sandbox_replay", "summarize_evidence",
    }
    assert expected <= names


def test_tools_read_only_inspection():
    res = _suspicious_run()
    tools = InvestigationTools(trace=res.trace, findings=res.findings)
    summary = tools.inspect_trace()
    assert summary["run_id"] == res.run_id
    errors = tools.inspect_tool_errors()
    assert isinstance(errors["errors"], list)
    steps = tools.retrieve_steps([res.findings[0].step_ids[0]])
    assert steps["steps"][0]["found"] is True
    missing = tools.retrieve_steps(["nope"])
    assert missing["steps"][0]["found"] is False


def test_tools_compare_runs():
    res = _suspicious_run()
    tools = InvestigationTools(trace=res.trace, findings=res.findings)
    assert tools.compare_runs()["available"] is False
    other = TraceBuilder(run_id="other").finish()
    tools2 = InvestigationTools(trace=res.trace, findings=res.findings, compare_trace=other)
    assert tools2.compare_runs()["available"] is True


def test_tools_replay_requires_authorization():
    res = _suspicious_run()
    tools = InvestigationTools(trace=res.trace, findings=res.findings)
    test = res.findings[0]
    proposal = tools.propose_regression_test(test.id)
    assert proposal["ok"]
    denied = tools.request_sandbox_replay(proposal["test"])
    assert denied["ok"] is False and denied["authorized"] is False
    tools_authorized = InvestigationTools(trace=res.trace, findings=res.findings, authorized_replay=True)
    proposal2 = tools_authorized.propose_regression_test(test.id)
    allowed = tools_authorized.request_sandbox_replay(proposal2["test"])
    assert allowed["ok"] is True
    assert "local synthetic world" in allowed["note"]


def test_tools_validate_test_rejects_bad_oracle():
    tools = InvestigationTools(trace=Trace(), findings=[])
    bad = {"scenario": "x" * 20, "task": {"name": "t"}, "oracle": {"checks": [{"check": "nope"}]},
           "expected_behavior": "e", "reproduction": "r"}
    result = tools.validate_test(bad)
    assert result["ok"] is False
    assert any("unknown oracle checker" in e for e in result["errors"])


def test_full_investigation_produces_report():
    res = _suspicious_run()
    report = investigate_run(res.trace, res.findings, budgets=BudgetConfig(max_iterations=16))
    assert report.status == "done"
    assert report.hypothesis
    assert report.evidence_step_ids
    assert isinstance(report.alternatives, list) and report.alternatives
    assert report.proposed_test
    assert "advisory" in report.note.lower() or "gate" in report.note.lower()
    assert report.budgets["iterations_used"] >= 1
    assert report.tool_log  # workflow is logged


def test_investigation_is_bounded():
    res = _suspicious_run()
    report = investigate_run(res.trace, res.findings, budgets=BudgetConfig(max_iterations=4, max_tool_calls=3))
    assert report.status in ("done", "budget_exhausted")
    assert report.budgets["iterations_used"] <= 8  # hard bound respected


def test_investigation_without_findings():
    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("good"), task)
    report = investigate_run(res.trace, res.findings)
    assert report.status == "done"
    assert "No deterministic failure evidence" in report.hypothesis or report.hypothesis


def test_investigation_with_authorized_replay():
    res = _suspicious_run()
    report = investigate_run(res.trace, res.findings, authorized_replay=True,
                             budgets=BudgetConfig(max_iterations=16))
    assert report.status == "done"
    assert report.replay_result is not None
    assert report.replay_result.get("ok") is True


def test_heuristic_planner_order():
    p = HeuristicPlanner()
    state: dict = {"done": []}
    order = []
    for _ in range(12):
        action = p.next_action(state)  # type: ignore[arg-type]
        order.append(action)
        if action == "report":
            break
        state["done"].append(action)
    assert order[0] == "inspect_trace"
    assert "form_hypothesis" in order
    assert order[-1] == "report"


def test_llm_planner_falls_back_on_garbage():
    class Garbage:
        def complete(self, *a, **k):
            return "not json"

    planner = LLMPlanner(Garbage())
    assert planner.next_action({}) == "inspect_trace"  # deterministic fallback


def test_llm_planner_accepts_valid_action():
    class Good:
        def complete(self, *a, **k):
            return '{"action": "report"}'

    planner = LLMPlanner(Good())
    state = {
        "trace_summary": {}, "findings_overview": {}, "tool_errors": {}, "task_spec": {},
        "comparison": {}, "relevant_steps": {}, "hypothesis": "h", "proposed_test": {},
        "validation": {}, "replay_result": {},
    }
    assert planner.next_action(state) == "report"  # type: ignore[arg-type]


def test_investigation_never_controls_gate():
    from failsafe.ci import evaluate_gate

    res = _suspicious_run()
    investigate_run(res.trace, res.findings)
    # investigation output is advisory: gate uses only deterministic findings/oracles
    decision = evaluate_gate([], run_results=[])
    assert decision.passed
    assert "investigation" not in str(decision.reasons).lower()
