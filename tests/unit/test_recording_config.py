"""Unit tests for the trace recorder and configuration."""

from __future__ import annotations

import json

import pytest

from failsafe.config import FailsafeConfig, load_config
from failsafe.recording import TraceRecorder
from failsafe.schemas import RunStatus, StepKind, StepStatus, TaskSpec


def test_recorder_basic_flow(tmp_path):
    rec = TraceRecorder(agent_name="agent", agent_version="1.0", task=TaskSpec(name="t"))
    tc = rec.record_tool_call("db_get", {"record_id": "u-1"})
    rec.record_tool_result(tc, {"plan": "free"})
    rec.record_response("The plan is free.")
    trace = rec.finish(status=RunStatus.SUCCESS)
    assert len(trace.steps) == 3
    assert trace.metadata.agent_name == "agent"
    assert trace.metadata.status == RunStatus.SUCCESS
    assert trace.metadata.ended_at is not None
    p = rec.save(tmp_path / "t.json")
    data = json.loads(p.read_text())
    assert data["metadata"]["run_id"] == trace.metadata.run_id


def test_recorder_span_context_manager():
    rec = TraceRecorder(agent_name="a")
    with rec.span(StepKind.PLANNING, "think") as step:
        rec.record_llm("llm", "prompt", "response")
    assert step.end_timestamp is not None
    assert rec.trace.steps[1].parent_id == step.id


def test_recorder_span_records_exceptions():
    rec = TraceRecorder(agent_name="a")
    with pytest.raises(ValueError):
        with rec.span(StepKind.PLANNING, "think"):
            raise ValueError("boom")
    step = rec.trace.steps[0]
    assert step.status == StepStatus.ERROR
    assert step.error_message == "boom"


def test_recorder_redacts_secrets():
    rec = TraceRecorder(agent_name="a", redact=True)
    tc = rec.record_tool_call("login", {"api_key": "sk-abcdefghijklmnop123456"})
    assert tc.arguments["api_key"] == "***REDACTED***"


def test_recorder_fault_log():
    from failsafe.schemas import FaultInjectionRecord, FaultKind

    rec = TraceRecorder(agent_name="a")
    rec.record_fault_injection(FaultInjectionRecord(kind=FaultKind.TIMEOUT, tool_name="t"))
    trace = rec.finish()
    assert trace.fault_injections[0].run_id == trace.metadata.run_id


# ---- config -------------------------------------------------------------

def test_default_config():
    cfg = FailsafeConfig()
    assert cfg.db_url.startswith("sqlite")
    assert cfg.detectors.retry_budget == 3
    assert cfg.ci.mode == "strict"
    assert cfg.llm_enabled is False


def test_load_config_from_file(tmp_path):
    p = tmp_path / "failsafe.json"
    p.write_text(json.dumps({"detectors": {"retry_budget": 7}, "llm_enabled": False}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.detectors.retry_budget == 7


def test_load_config_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILSAFE_DETECTORS__RETRY_BUDGET", "11")
    monkeypatch.setenv("FAILSAFE_LLM_ENABLED", "true")
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.detectors.retry_budget == 11
    assert cfg.llm_enabled is True


def test_load_config_explicit_overrides_win(tmp_path):
    p = tmp_path / "failsafe.json"
    p.write_text(json.dumps({"detectors": {"retry_budget": 7}}), encoding="utf-8")
    cfg = load_config(p, overrides={"detectors": {"retry_budget": 9}})
    assert cfg.detectors.retry_budget == 9


def test_config_rejects_unknown_keys(tmp_path):
    from pydantic import ValidationError

    p = tmp_path / "failsafe.json"
    p.write_text(json.dumps({"bogus_section": {}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(p)


def test_config_coercion(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILSAFE_INJECTION__MAX_INJECTIONS", "3")
    monkeypatch.setenv("FAILSAFE_INJECTION__SEED", "42")
    cfg = load_config(tmp_path / "none.json")
    assert cfg.injection.max_injections == 3
    assert cfg.injection.seed == 42
