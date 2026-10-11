"""Regression tests generated from observed Failsafe failures.

Each test encodes a defect that was actually observed during development and
must not come back.  These are ordinary pytest tests (deterministic, no LLM).

Source: Failsafe development log — see docs/implementation_status.md.
"""

from __future__ import annotations

from failsafe.agents.sample import get_agent
from failsafe.evaluation import execute_task, get_builtin_tasks
from failsafe.recording import TraceRecorder
from failsafe.schemas import RunStatus, TaskSpec


def test_recorder_tool_call_steps_carry_tool_payloads():
    """REGRESSION (defect #1): TraceRecorder.add_step rejected tool_call/tool_result
    kwargs, silently corrupting every recorded tool step (the TypeError was
    swallowed by the runner and recorded as a run error)."""
    rec = TraceRecorder(agent_name="a", task=TaskSpec(name="t"))
    tc = rec.record_tool_call("db_get", {"record_id": "u-1"})
    assert tc.step_id is not None
    step = rec.trace.step_by_id(tc.step_id)
    assert step is not None
    assert step.tool_call is not None and step.tool_call.name == "db_get"
    tr = rec.record_tool_result(tc, {"ok": True})
    rstep = rec.trace.step_by_id(tr.step_id)
    assert rstep is not None and rstep.tool_result is not None


def test_recorder_finish_attaches_metadata():
    """REGRESSION (defect #2): Pydantic copied RunMetadata on Trace construction,
    so recorder.finish() updated a detached object and the trace metadata kept
    status=unknown / final_response=None."""
    rec = TraceRecorder(agent_name="a")
    rec.record_response("done")
    trace = rec.finish(status=RunStatus.SUCCESS, final_response="done")
    assert trace.metadata.status == RunStatus.SUCCESS
    assert trace.metadata.final_response == "done"
    assert trace.metadata.ended_at is not None


def test_unsupported_success_negation_not_success_claim():
    """REGRESSION (defect #3): 'could not be completed' matched the success lexicon
    ('completed') and produced a false success-claim finding on honest failure
    reports."""
    from failsafe.detectors.unsupported_success import claims_success

    assert claims_success("The update could not be completed. No changes were made.") is False
    assert claims_success("Failed to update u-9999.") is False
    assert claims_success("Task completed successfully!") is True
    task = next(t for t in get_builtin_tasks() if t.id == "task-unrecoverable-fault")
    res = execute_task(get_agent("good"), task)
    assert res.oracle.success
    ids = {f.detector_id for f in res.findings}
    assert "unsupported_success" not in ids


def test_good_agent_recovers_and_gate_passes():
    """REGRESSION (defect #4): recovery was counted for safe-handling runs
    (explicit failure reports), conflating two distinct reliability metrics."""
    task = next(t for t in get_builtin_tasks() if t.id == "task-recoverable-fault")
    res = execute_task(get_agent("good"), task)
    assert res.recovered and not res.safe_handled
    task2 = next(t for t in get_builtin_tasks() if t.id == "task-unrecoverable-fault")
    res2 = execute_task(get_agent("good"), task2)
    assert res2.safe_handled and not res2.recovered

    from failsafe.ci import evaluate_gate

    decision = evaluate_gate(res.findings + res2.findings, run_results=[res, res2])
    assert decision.passed  # INFO/LOW findings and unresolved markers never block


def test_markdown_report_not_html_escaped():
    """REGRESSION (defect #5): the Markdown renderer HTML-escaped evidence quotes
    (&#34; instead of \")."""
    from failsafe.reporting import build_report_payload, render_markdown

    task = next(t for t in get_builtin_tasks() if t.id == "task-db-lookup")
    res = execute_task(get_agent("buggy"), task)
    payload = build_report_payload(title="R", trace=res.trace, findings=res.findings)
    md = render_markdown(payload)
    assert "&#34;" not in md
    assert "&quot;" not in md


def test_repeated_actions_retry_after_failure_not_escalated():
    """REGRESSION (defect #6): legitimate retries after an injected fault were
    escalated as repeated actions with the same severity as genuine loops."""
    task = next(t for t in get_builtin_tasks() if t.id == "task-recoverable-fault")
    res = execute_task(get_agent("good"), task)
    reps = [f for f in res.findings if f.detector_id == "repeated_actions"]
    for f in reps:
        assert f.severity.value == "info"
        assert f.metrics["non_retry_count"] < f.metrics["count"]
