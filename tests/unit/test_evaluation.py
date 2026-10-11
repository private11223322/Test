"""Unit tests for oracle checks, tasks, and metrics."""

from __future__ import annotations

import pytest

from failsafe.agents.sample import FlakyAgent, GoodAgent, get_agent
from failsafe.evaluation import (
    SyntheticWorld,
    checker_names,
    compute_run_metrics,
    confusion_matrix,
    detector_metrics,
    execute_task,
    get_builtin_tasks,
    live_run_variability,
    make_demo_world,
    precision_recall_f1,
    replay_stability,
    run_checks,
)
from failsafe.evaluation.metrics import RunResult
from failsafe.evaluation.world import AuditEntry
from failsafe.schemas import (
    DetectorFinding,
    OracleCheck,
    OracleOutcome,
    RunStatus,
    Trace,
)
from tests.conftest import TraceBuilder

# ---- world --------------------------------------------------------------

def test_world_db_ops():
    w = SyntheticWorld(records={"a": {"id": "a", "x": 1}})
    assert w.db_get("a")["x"] == 1
    w.db_update("a", {"x": 2})
    assert w.db_get("a")["x"] == 2
    with pytest.raises(KeyError):
        w.db_get("missing")
    w.db_insert("b", {"id": "b"})
    w.db_delete("b")
    assert "b" not in w.db
    assert w.mutation_count("a", "update") == 1


def test_world_search_controlled():
    w = make_demo_world()
    hits = w.search("refund policy")
    assert hits and hits[0]["id"] == "doc-1"
    assert w.search("nonexistent term xyz") == []


def test_world_isolated_between_calls():
    w1 = make_demo_world()
    w2 = make_demo_world()
    w1.db_update("u-1001", {"plan": "pro"})
    assert w2.db["u-1001"]["plan"] == "free"


def test_world_audit_log():
    w = make_demo_world()
    w.db_update("u-1001", {"plan": "pro"})
    w.db_update("u-1001", {"plan": "enterprise"})
    assert len(w.audit_log) == 2
    assert all(isinstance(e, AuditEntry) for e in w.audit_log)


# ---- oracle checkers ----------------------------------------------------

def test_checkers_registered():
    names = checker_names()
    for expected in ("world_record_equals", "response_contains", "custom", "no_duplicate_mutation"):
        assert expected in names


def test_world_record_equals_check():
    w = make_demo_world()
    t = Trace()
    out = run_checks(w, t, [{"check": "world_record_equals", "id": "u-1001", "fields": {"plan": "free"}}])
    assert out.success
    out2 = run_checks(w, t, [{"check": "world_record_equals", "id": "u-1001", "fields": {"plan": "pro"}}])
    assert not out2.success


def test_unknown_checker_fails_loudly():
    out = run_checks(make_demo_world(), Trace(), [{"check": "no_such_checker"}])
    assert not out.success
    assert "Unknown checker" in out.checks[0].detail


def test_crashing_checker_is_failed_check():
    # path traversal into a non-container yields None; op gt on None raises -> failed check
    out = run_checks(make_demo_world(), Trace(), [{"check": "custom", "path": ["db", "u-1001", "plan"], "op": "gt", "value": 1}])
    assert not out.success


def test_custom_check_ops():
    w = make_demo_world()
    t = Trace()
    ok = run_checks(w, t, [{"check": "custom", "path": ["db", "u-1001", "plan"], "op": "eq", "value": "free"}])
    assert ok.success
    missing = run_checks(w, t, [{"check": "custom", "path": ["db", "nope"], "op": "exists"}])
    assert not missing.success
    exists = run_checks(w, t, [{"check": "custom", "path": ["db", "u-1001"], "op": "exists"}])
    assert exists.success


def test_no_duplicate_mutation_check():
    w = make_demo_world()
    t = Trace()
    w.db_update("u-1001", {"plan": "pro"})
    ok = run_checks(w, t, [{"check": "no_duplicate_mutation"}])
    assert ok.success
    w.db_update("u-1001", {"plan": "pro"})  # identical duplicate
    bad = run_checks(w, t, [{"check": "no_duplicate_mutation"}])
    assert not bad.success


def test_response_checks_use_final_response():
    t = Trace()
    t.metadata.final_response = "The answer is 42."
    ok = run_checks(make_demo_world(), t, [{"check": "response_contains", "text": "42"}])
    assert ok.success
    no = run_checks(make_demo_world(), t, [{"check": "response_not_contains", "text": "43"}])
    assert no.success


# ---- runner / tasks -----------------------------------------------------

def test_builtin_tasks_count_and_categories():
    tasks = get_builtin_tasks()
    assert len(tasks) >= 10
    categories = {t.category for t in tasks}
    for needed in ("database_lookup", "multi_step", "search", "record_update", "recovery", "safe_failure", "stale_data", "duplicate_mutation"):
        assert needed in categories


def test_execute_task_good_agent():
    task = get_builtin_tasks()[0]
    res = execute_task(GoodAgent(), task)
    assert res.oracle.success
    assert res.trace.metadata.run_id
    assert res.runtime_ms >= 0


def test_execute_task_buggy_agent_silent_failure():
    task = next(t for t in get_builtin_tasks() if t.id == "task-update-record")
    res = execute_task(get_agent("buggy"), task)
    assert not res.oracle.success
    assert res.silent_failure


def test_execute_task_fault_injection_recovery():
    task = next(t for t in get_builtin_tasks() if t.id == "task-recoverable-fault")
    res = execute_task(GoodAgent(), task)
    assert res.faults_injected >= 1
    assert res.oracle.success
    assert res.recovered
    assert res.trace.fault_injections  # injections logged


def test_execute_task_safe_handling_not_recovery():
    task = next(t for t in get_builtin_tasks() if t.id == "task-unrecoverable-fault")
    res = execute_task(GoodAgent(), task)
    assert res.oracle.success
    assert res.safe_handled
    assert not res.recovered


def test_execute_task_error_recorded():
    class ExplodingAgent:
        name = "boom"

        def run(self, task, tools, recorder):
            raise RuntimeError("agent crashed")

    res = execute_task(ExplodingAgent(), get_builtin_tasks()[0])
    assert res.error and "agent crashed" in res.error
    assert res.trace.metadata.status == RunStatus.ERROR


def test_flaky_agent_ignored_error_detected():
    task = next(t for t in get_builtin_tasks() if t.id == "task-recoverable-fault")
    res = execute_task(FlakyAgent(), task)
    ids = {f.detector_id for f in res.findings}
    assert "ignored_tool_errors" in ids
    assert res.silent_failure


# ---- metrics ------------------------------------------------------------

def _fake_result(success=True, faults_injected=0, faults_scheduled=0, reported_failure=False,
                 claimed_success=True, run_id="r") -> RunResult:
    builder = TraceBuilder(run_id=run_id)
    if claimed_success:
        builder.response("Task completed successfully.")
    else:
        builder.response("Failed to do it.")
    trace = builder.finish(status=RunStatus.SUCCESS if claimed_success else RunStatus.FAILURE)
    oracle = OracleOutcome(run_id=run_id, task_id="t", task_name="t", success=success,
                           checks=[OracleCheck(name="c", passed=success)])
    return RunResult(run_id=run_id, task_id="t", task_name="t", trace=trace, oracle=oracle,
                     faults_injected=faults_injected, faults_scheduled=faults_scheduled,
                     agent_reported_failure=reported_failure)


def test_compute_run_metrics_rates():
    results = [
        _fake_result(success=True),  # plain success
        _fake_result(success=False, claimed_success=True, run_id="r2"),  # silent failure
        # safe handling: agent reported failure and the oracle verified that
        # reporting failure was the correct outcome (oracle success=True)
        _fake_result(success=True, claimed_success=False, reported_failure=True, run_id="r3"),
        _fake_result(success=True, faults_injected=1, faults_scheduled=1, run_id="r4"),  # recovered
    ]
    m = compute_run_metrics(results)
    assert m.get("task_success_rate").value == 0.75
    assert m.get("verified_silent_failure_rate").value == 0.25
    assert m.get("fault_recovery_rate").value == 1.0  # 1 recovered of 1 injected
    assert m.get("safe_handling_rate").value == 0.25


def test_injection_failure_never_counts_as_recovery():
    results = [_fake_result(success=True, faults_injected=0, faults_scheduled=2, run_id="r1")]
    m = compute_run_metrics(results)
    assert m.get("fault_recovery_rate").denominator == 0
    assert m.get("fault_recovery_rate").value is None
    assert m.get("injection_failure_rate").value == 1.0


def test_empty_results():
    m = compute_run_metrics([])
    assert m.values == []


def test_confusion_matrix():
    cm = confusion_matrix([True, True, False, False], [True, False, True, False])
    assert cm == {"tn_negative": 1, "fp_positive": 1, "fn_positive": 1, "tp_positive": 1}


def test_precision_recall_f1():
    prf = precision_recall_f1(tp=2, fp=1, fn=1)
    assert prf["precision"] == pytest.approx(2 / 3)
    assert prf["recall"] == pytest.approx(2 / 3)
    assert prf["f1"] == pytest.approx(2 / 3)
    assert precision_recall_f1(0, 0, 0)["f1"] is None


def test_detector_metrics_perfect():
    f = DetectorFinding(run_id="r", detector_id="d", category="repeated_actions",
                        summary="x", evidence=[])
    f.evidence = []
    from failsafe.schemas import Evidence
    f = DetectorFinding(run_id="r", detector_id="d", category="repeated_actions", summary="loop",
                        evidence=[Evidence(kind="k", step_ids=["s1"])])
    m = detector_metrics({"r": [f]}, {"r": [{"category": "repeated_actions", "step_ids": ["s1"]}]})
    assert m.get("detector_precision").value == 1.0
    assert m.get("detector_recall").value == 1.0


def test_detector_metrics_false_positive():
    from failsafe.schemas import Evidence
    f = DetectorFinding(run_id="r", detector_id="d", category="c", summary="s",
                        evidence=[Evidence(kind="k", step_ids=["s1"])])
    m = detector_metrics({"r": [f]}, {"r": []})
    assert m.get("detector_precision").value == 0.0
    assert m.get("detector_recall").value is None
    assert m.sample_counts["fp"] == 1


def test_replay_stability():
    a = [_fake_result(success=True, run_id="r1"), _fake_result(success=False, run_id="r2")]
    b = [_fake_result(success=True, run_id="r1"), _fake_result(success=True, run_id="r2")]
    stab = replay_stability(a, b)
    assert stab.numerator == 1 and stab.denominator == 2


def test_live_run_variability():
    v = live_run_variability([True, True, False, True])
    assert v.value == 0.75
    assert live_run_variability([]).value is None
