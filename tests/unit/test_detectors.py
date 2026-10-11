"""Unit tests for the deterministic failure detectors.

Each detector has positive (must fire), negative (must not fire), boundary
(threshold), and ambiguous (labelled, not asserted) cases.
"""

from __future__ import annotations

import pytest

from failsafe.config import DetectorConfig
from failsafe.detectors import (
    BUILTIN_DETECTOR_IDS,
    registry,
    run_detectors,
)
from failsafe.schemas import FindingStatus, Severity, TaskSpec

# ---- registry -----------------------------------------------------------

def test_all_builtin_detectors_registered():
    assert set(BUILTIN_DETECTOR_IDS) == set(registry.ids())


def test_registry_create_unknown():
    with pytest.raises(KeyError):
        registry.create("no-such-detector")


def test_registry_rejects_duplicate():
    from failsafe.detectors.base import Detector

    class Dup(Detector):
        id = "repeated_actions"
        category = "x"

        def detect(self, trace):
            return []

    with pytest.raises(ValueError):
        registry.register(Dup)


def test_run_detectors_deterministic(builder):
    t = builder()
    for _ in range(4):
        t.call("db_get", {"record_id": "u-1"})
    t.finish()
    a = run_detectors(t.trace)
    b = run_detectors(t.trace)
    assert [f.summary for f in a] == [f.summary for f in b]
    assert [f.step_ids for f in a] == [f.step_ids for f in b]


def test_clean_trace_no_findings(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"}, output={"plan": "free"})
    t.response("The plan is free.")
    t.finish()
    assert run_detectors(t.trace) == []


# ---- A: repeated actions ------------------------------------------------

def test_repeated_actions_positive(builder):
    t = builder()
    for _ in range(4):
        t.call("db_get", {"record_id": "u-1"})
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "repeated_actions"]
    assert len(findings) == 1
    f = findings[0]
    assert len(f.step_ids) == 4  # exact step IDs of the four tool calls
    assert f.metrics["count"] == 4
    assert f.severity in (Severity.MEDIUM, Severity.HIGH)


def test_repeated_actions_negative_single_call(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"})
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "repeated_actions"] == []


def test_repeated_actions_negative_different_args(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"})
    t.call("db_get", {"record_id": "u-2"})
    t.call("db_get", {"record_id": "u-3"})
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "repeated_actions"] == []


def test_repeated_actions_normalized_args(builder):
    t = builder()
    t.call("search", {"query": "hello   world"})
    t.call("search", {"query": "hello world"})
    t.call("search", {"query": "hello world"})
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "repeated_actions"]
    assert len(findings) == 1
    assert findings[0].metrics["count"] == 3


def test_repeated_actions_boundary_two_vs_min(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"})
    t.call("db_get", {"record_id": "u-1"})
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "repeated_actions"]
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO  # below min_count=3
    assert findings[0].metrics["non_retry_count"] == 2


def test_repeated_actions_retries_excluded_from_escalation(builder):
    # identical retry after failure is a retry chain, not a repeated action
    t = builder()
    tc1 = t.tool_call("db_update", {"record_id": "u-1", "fields": {"a": 1}})
    t.tool_result(tc1, success=False, error_message="boom")
    tc2 = t.tool_call("db_update", {"record_id": "u-1", "fields": {"a": 1}})
    t.tool_result(tc2, success=True)
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "repeated_actions"]
    # reported as evidence-level info with non_retry_count == 1
    assert findings[0].metrics["non_retry_count"] == 1
    assert findings[0].severity == Severity.INFO


def test_repeated_actions_config_threshold(builder):
    t = builder()
    for _ in range(3):
        t.call("db_get", {"record_id": "u-1"})
    t.finish()
    findings = run_detectors(t.trace, DetectorConfig(repeat_min_count=3))
    f = [x for x in findings if x.detector_id == "repeated_actions"][0]
    assert f.severity == Severity.MEDIUM
    findings2 = run_detectors(t.trace, DetectorConfig(repeat_min_count=5))
    f2 = [x for x in findings2 if x.detector_id == "repeated_actions"][0]
    assert f2.severity == Severity.INFO


# ---- B: excessive retries ----------------------------------------------

def test_excessive_retries_positive(builder):
    t = builder()
    for _ in range(5):
        t.call("db_get", {"record_id": "u-1"}, success=False, error_message="down")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "excessive_retries"]
    assert len(findings) == 1
    assert findings[0].metrics["consecutive_failures"] == 5
    assert findings[0].severity == Severity.HIGH


def test_excessive_retries_negative_within_budget(builder):
    t = builder()
    for _ in range(3):
        t.call("db_get", {"record_id": "u-1"}, success=False, error_message="down")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "excessive_retries"] == []


def test_excessive_retries_boundary(builder):
    t = builder()
    for _ in range(4):
        t.call("db_get", {"record_id": "u-1"}, success=False, error_message="down")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "excessive_retries"]
    assert len(findings) == 1  # 4 > budget 3


def test_excessive_retries_resets_on_success(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"}, success=False)
    t.call("db_get", {"record_id": "u-1"}, success=False)
    t.call("db_get", {"record_id": "u-1"}, success=True)
    t.call("db_get", {"record_id": "u-1"}, success=False)
    t.call("db_get", {"record_id": "u-1"}, success=False)
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "excessive_retries"] == []


def test_excessive_retries_configurable_budget(builder):
    t = builder()
    for _ in range(3):
        t.call("op", {}, success=False)
    t.finish()
    assert [f for f in run_detectors(t.trace, DetectorConfig(retry_budget=3)) if f.detector_id == "excessive_retries"] == []
    assert [f for f in run_detectors(t.trace, DetectorConfig(retry_budget=2)) if f.detector_id == "excessive_retries"] != []


# ---- C: ignored tool errors ---------------------------------------------

def test_ignored_tool_errors_positive(builder):
    t = builder(task=TaskSpec(name="t", tool_policy={"require_failure_report": True}))
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.response("Everything is fine!")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "ignored_tool_errors"]
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH
    assert findings[0].evidence[0].step_ids  # exact step IDs


def test_ignored_tool_errors_negative_retry(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.call("db_get", {"record_id": "u-1"}, success=True)
    t.response("Done.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "ignored_tool_errors"] == []


def test_ignored_tool_errors_negative_fallback(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.call("search", {"query": "u-1"}, success=True)
    t.response("Found it via search.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "ignored_tool_errors"] == []


def test_ignored_tool_errors_negative_explicit_report(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.response("I could not complete the lookup: the database is down. Task failed.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "ignored_tool_errors"] == []


def test_ignored_tool_errors_policy_off(builder):
    t = builder(task=TaskSpec(name="t", tool_policy={"require_failure_report": False}))
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.response("Whatever.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "ignored_tool_errors"] == []


def test_ignored_tool_errors_ambiguous_case(builder):
    # error then unrelated work with no comment: flagged (rule-triggered, not verified failure)
    t = builder()
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.llm("thinking", "hmm")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "ignored_tool_errors"]
    assert len(findings) == 1
    assert findings[0].status == FindingStatus.RULE_TRIGGERED


# ---- D: premature termination -------------------------------------------

def test_premature_termination_positive(builder):
    task = TaskSpec(name="t", required_outcomes=[{"type": "tool_called", "tool": "db_update"}])
    t = builder(task=task)
    t.call("db_get", {"record_id": "u-1"})
    t.response("Done.")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "premature_termination"]
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH


def test_premature_termination_negative(builder):
    task = TaskSpec(name="t", required_outcomes=[{"type": "tool_called", "tool": "db_update"}])
    t = builder(task=task)
    t.call("db_update", {"record_id": "u-1"}, success=True)
    t.response("Updated.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "premature_termination"] == []


def test_premature_termination_no_task_spec(builder):
    t = builder()
    t.response("Done without doing anything.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "premature_termination"] == []


def test_premature_termination_multiple_unmet(builder):
    task = TaskSpec(name="t", required_outcomes=[
        {"type": "tool_called", "tool": "db_update"},
        {"type": "response_contains", "text": "shipped"},
    ])
    t = builder(task=task)
    t.response("Nothing happened.")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "premature_termination"]
    assert len(findings) == 2


def test_premature_termination_response_contains(builder):
    task = TaskSpec(name="t", required_outcomes=[{"type": "response_contains", "text": "42"}])
    t = builder(task=task)
    t.response("The answer is 42.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "premature_termination"] == []


# ---- E: unsupported success claims --------------------------------------

def test_unsupported_success_positive(builder):
    task = TaskSpec(name="t", required_outcomes=[{"type": "tool_called", "tool": "db_update"}])
    t = builder(task=task)
    t.response("Task completed successfully! All updates applied.", claimed_success=True)
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "unsupported_success"]
    assert len(findings) == 1
    assert findings[0].status == FindingStatus.RULE_TRIGGERED
    assert findings[0].severity == Severity.HIGH


def test_unsupported_success_negative_honest_failure(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.response("Failed to complete the task: the tool is down.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "unsupported_success"] == []


def test_unsupported_success_negation_context(builder):
    # "could not be completed" must not be read as a success claim
    t = builder()
    t.response("The update could not be completed. No changes were made.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "unsupported_success"] == []


def test_unsupported_success_ambiguous_after_errors(builder):
    task = TaskSpec(name="t", required_outcomes=[{"type": "response_contains", "text": "yes"}])
    t = builder(task=task)
    t.call("db_get", {"record_id": "u-1"}, success=False, error_message="boom")
    t.response("yes, done")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "unsupported_success"]
    # required outcome satisfied but unrecovered tool failure -> UNRESOLVED, not asserted
    assert len(findings) == 1
    assert findings[0].status == FindingStatus.UNRESOLVED
    assert findings[0].severity == Severity.LOW


def test_unsupported_success_recovered_failure_not_flagged(builder):
    t = builder()
    tc1 = t.tool_call("db_update", {"record_id": "u-1", "fields": {"a": 1}})
    t.tool_result(tc1, success=False, error_message="transient")
    tc2 = t.tool_call("db_update", {"record_id": "u-1", "fields": {"a": 1}})
    t.tool_result(tc2, success=True)
    t.response("Updated successfully.")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "unsupported_success"] == []


def test_unsupported_success_no_response(builder):
    t = builder()
    t.call("db_get", {"record_id": "u-1"})
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "unsupported_success"] == []


# ---- F: lost / repeated context -----------------------------------------

def test_lost_context_repeated_request(builder):
    t = builder()
    t.llm("What is the plan for u-1?", "free")
    t.llm("What is the plan for u-1?", "free")
    t.response("The plan is free.")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "lost_context"]
    assert len(findings) == 1
    assert findings[0].evidence[0].kind == "repeated_request"
    assert len(findings[0].step_ids) == 2


def test_lost_context_negative_distinct_prompts(builder):
    t = builder()
    t.llm("What is the plan for u-1?", "free")
    t.llm("What is the plan for u-2?", "pro")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "lost_context"] == []


def test_lost_context_boundary_min_count(builder):
    t = builder()
    t.llm("same question", "a")
    t.finish()
    assert [f for f in run_detectors(t.trace) if f.detector_id == "lost_context"] == []
    t.llm("same question", "b")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "lost_context"]
    assert len(findings) == 1  # context_repeat_min_count=2 boundary


def test_lost_context_ambiguous_reask(builder):
    t = builder()
    t.llm("please look up the refund policy", "14 days")
    t.llm("could you look up the refund policy again?", "")
    t.finish()
    findings = [f for f in run_detectors(t.trace) if f.detector_id == "lost_context"]
    ambiguous = [f for f in findings if f.evidence[0].kind == "possible_context_loss"]
    if ambiguous:
        assert ambiguous[0].status == FindingStatus.UNRESOLVED
        assert ambiguous[0].severity == Severity.LOW


# ---- detector interface contract ----------------------------------------

def test_detector_interface_fields():
    for did in registry.ids():
        d = registry.create(did)
        assert d.id == did
        assert d.category
        assert d.description
        assert isinstance(d.known_limitations, list)
        assert d.known_limitations  # documented limitations required


def test_detector_config_isolated_per_instance():
    d1 = registry.create("excessive_retries", DetectorConfig(retry_budget=1))
    d2 = registry.create("excessive_retries", DetectorConfig(retry_budget=9))
    assert d1.config.retry_budget == 1
    assert d2.config.retry_budget == 9
