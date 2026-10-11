"""Unit tests for core trace schemas."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from failsafe.schemas import (
    DetectorFinding,
    Evidence,
    FaultInjectionRecord,
    FaultKind,
    RunMetadata,
    RunStatus,
    Step,
    StepKind,
    StepStatus,
    TaskSpec,
    ToolCall,
    ToolResult,
    Trace,
)


def test_step_defaults_preserve_missing():
    s = Step(kind=StepKind.OTHER)
    assert s.status == StepStatus.UNKNOWN or s.status == StepStatus.OK
    assert s.tool_call is None
    assert s.tool_result is None
    assert s.error_message is None  # missing info preserved as None


def test_step_duration():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    s = Step(kind=StepKind.LLM, timestamp=start, end_timestamp=start + timedelta(milliseconds=250))
    assert s.duration_ms == pytest.approx(250.0)


def test_step_duration_missing_end():
    s = Step(kind=StepKind.LLM)
    assert s.duration_ms is None


def test_trace_lookup_helpers():
    t = Trace()
    s1 = Step(id="a", kind=StepKind.TOOL_CALL, tool_call=ToolCall(id="tc1", name="db_get", arguments={"x": 1}))
    s2 = Step(id="b", kind=StepKind.TOOL_RESULT, tool_result=ToolResult(tool_call_id="tc1", success=True))
    s3 = Step(id="c", kind=StepKind.LLM, parent_id="a")
    t.steps = [s3, s1, s2]
    assert t.step_by_id("a") is s1
    assert t.step_by_id("nope") is None
    assert len(t.tool_calls()) == 1
    assert len(t.tool_results()) == 1
    assert t.children_of("a")[0].id == "c"
    assert [s.id for s in t.ordered_steps()] == ["a", "b", "c"]


def test_strict_model_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        Step.model_validate({"kind": "llm", "bogus_key": 1})


def test_loose_model_allows_extra_metadata():
    meta = RunMetadata.model_validate({"run_id": "r1", "custom_field": "kept"})
    assert meta.run_id == "r1"
    assert meta.model_extra.get("custom_field") == "kept"


def test_run_status_values():
    assert RunStatus.SUCCESS.value == "success"
    m = RunMetadata(status="success")
    assert m.status == RunStatus.SUCCESS


def test_finding_step_ids_unique_ordered():
    f = DetectorFinding(
        run_id="r", detector_id="d", category="c", summary="s",
        evidence=[
            Evidence(kind="k", step_ids=["s1", "s2"]),
            Evidence(kind="k2", step_ids=["s2", "s3"]),
        ],
    )
    assert f.step_ids == ["s1", "s2", "s3"]


def test_fault_injection_record_roundtrip():
    fi = FaultInjectionRecord(kind=FaultKind.RATE_LIMIT, tool_name="db_get", details={"retry_after": 1})
    dumped = fi.model_dump(mode="json")
    again = FaultInjectionRecord.model_validate(dumped)
    assert again.kind == FaultKind.RATE_LIMIT
    assert again.details["retry_after"] == 1


def test_tool_result_missing_output_is_none():
    tr = ToolResult()
    assert tr.output is None
    assert tr.success is True


def test_task_spec_roundtrip():
    task = TaskSpec(
        name="t", required_outcomes=[{"type": "tool_called", "tool": "db_get"}],
        tool_policy={"require_failure_report": True},
    )
    again = TaskSpec.model_validate(task.model_dump(mode="json"))
    assert again.required_outcomes[0]["tool"] == "db_get"
    assert again.tool_policy["require_failure_report"] is True


def test_trace_rejects_bad_types():
    with pytest.raises(ValidationError):
        Trace.model_validate({"steps": "not-a-list"})
