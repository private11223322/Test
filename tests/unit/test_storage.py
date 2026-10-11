"""Unit tests for the storage/repository layer."""

from __future__ import annotations

from failsafe.agents.regression import generate_regression_test
from failsafe.agents.sample import get_agent
from failsafe.evaluation import execute_task, get_builtin_tasks
from failsafe.schemas import (
    DetectorFinding,
    EvaluationRecord,
    Evidence,
    FindingStatus,
    Severity,
)
from failsafe.storage import Repository


def _repo(tmp_path):
    return Repository(f"sqlite:///{tmp_path}/test.db")


def test_save_and_load_trace(tmp_path):
    repo = _repo(tmp_path)
    task = get_builtin_tasks()[0]
    res = execute_task(get_agent("good"), task)
    run_id = repo.save_trace(res.trace)
    loaded = repo.get_trace(run_id)
    assert loaded is not None
    assert loaded.metadata.run_id == run_id
    assert len(loaded.steps) == len(res.trace.steps)


def test_list_runs_filtering(tmp_path):
    repo = _repo(tmp_path)
    for name in ("good", "buggy"):
        res = execute_task(get_agent(name), get_builtin_tasks()[0])
        repo.save_trace(res.trace)
    assert len(repo.list_runs()) == 2
    assert len(repo.list_runs(agent_name="good-agent")) == 1
    assert len(repo.list_runs(status="success")) == 2


def test_findings_roundtrip_and_status_update(tmp_path):
    repo = _repo(tmp_path)
    res = execute_task(get_agent("buggy"), get_builtin_tasks()[0])
    repo.save_trace(res.trace)
    repo.save_findings(res.findings)
    stored = repo.list_findings(run_id=res.run_id)
    assert len(stored) == len(res.findings)
    fid = stored[0].id
    repo.update_finding_status(fid, FindingStatus.CONFIRMED_BY_HUMAN.value)
    again = [f for f in repo.list_findings(run_id=res.run_id) if f.id == fid][0]
    assert again.status == FindingStatus.CONFIRMED_BY_HUMAN.value


def test_findings_filter_by_category(tmp_path):
    repo = _repo(tmp_path)
    f = DetectorFinding(
        run_id="r-x", detector_id="d", category="repeated_actions", summary="s",
        severity=Severity.MEDIUM, status=FindingStatus.RULE_TRIGGERED,
        evidence=[Evidence(kind="k", step_ids=["s1"])],
    )
    repo.save_findings([f])
    assert len(repo.list_findings(category="repeated_actions")) == 1
    assert len(repo.list_findings(category="other")) == 0


def test_oracle_roundtrip(tmp_path):
    repo = _repo(tmp_path)
    res = execute_task(get_agent("good"), get_builtin_tasks()[0])
    repo.save_trace(res.trace)
    repo.save_oracle(res.oracle)
    stored = repo.list_oracles(run_id=res.run_id)
    assert len(stored) == 1
    assert stored[0].success == res.oracle.success


def test_regression_test_roundtrip(tmp_path):
    repo = _repo(tmp_path)
    res = execute_task(get_agent("looping"), get_builtin_tasks()[0])
    repo.save_trace(res.trace)
    repo.save_findings(res.findings)
    tests = generate_regression_test(res.trace, res.findings[0]) if res.findings else None
    if tests is None:
        return
    repo.save_regression_test(tests)
    listed = repo.list_regression_tests()
    assert len(listed) == 1
    loaded = repo.get_regression_test(tests.id)
    assert loaded is not None and loaded.scenario == tests.scenario
    repo.update_regression_test_state(tests.id, "failing", {"note": 1})
    assert repo.list_regression_tests(state="failing")[0].last_result == {"note": 1}


def test_evaluations_and_reports(tmp_path):
    repo = _repo(tmp_path)
    repo.save_evaluation(EvaluationRecord(name="e", kind="suite", data={"x": 1}))
    assert len(repo.list_evaluations(kind="suite")) == 1
    rid = repo.save_report(None, "html", "/tmp/x.html")
    assert rid
    assert len(repo.list_reports()) == 1


def test_delete_run(tmp_path):
    repo = _repo(tmp_path)
    res = execute_task(get_agent("good"), get_builtin_tasks()[0])
    repo.save_trace(res.trace)
    repo.save_findings(res.findings)
    repo.save_oracle(res.oracle)
    repo.delete_run(res.run_id)
    assert repo.get_run(res.run_id) is None
    assert repo.list_findings(run_id=res.run_id) == []
