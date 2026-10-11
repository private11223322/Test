"""Unit tests for trace ingestion, redaction, and limits."""

from __future__ import annotations

import json

import pytest

from failsafe.ingestion import TraceSizeLimitExceeded, enforce_limits, redact_text, redact_value
from failsafe.ingestion.loader import (
    TraceValidationError,
    detect_format,
    ingest_trace_dict,
    load_trace,
    load_trace_bundle,
    validate_trace_file,
)
from failsafe.ingestion.normalize import normalize_openinference
from failsafe.schemas import StepKind

VALID_TRACE = {
    "metadata": {"run_id": "r-1", "agent_name": "agent", "status": "success"},
    "steps": [
        {"id": "s1", "kind": "tool_call", "name": "db_get",
         "tool_call": {"id": "tc1", "name": "db_get", "arguments": {"record_id": "u-1"}}},
        {"id": "s2", "kind": "tool_result", "name": "db_get",
         "tool_result": {"tool_call_id": "tc1", "name": "db_get", "success": True, "output": {"plan": "free"}}},
        {"id": "s3", "kind": "response", "name": "final", "output": "The plan is free."},
    ],
}


def test_detect_format_native():
    assert detect_format(VALID_TRACE) == "failsafe"


def test_detect_format_openinference():
    assert detect_format({"spans": [{"span_id": "1", "attributes": {}}]}) == "openinference"


def test_ingest_valid_trace():
    t = ingest_trace_dict(json.loads(json.dumps(VALID_TRACE)))
    assert t.metadata.run_id == "r-1"
    assert len(t.steps) == 3
    assert t.steps[0].kind == StepKind.TOOL_CALL
    # tool_call.step_id backfilled
    assert t.steps[0].tool_call.step_id == "s1"


def test_malformed_trace_gives_useful_errors():
    bad = {"metadata": {"run_id": 1}, "steps": [{"kind": "not-a-kind"}]}
    with pytest.raises(TraceValidationError) as exc:
        ingest_trace_dict(bad)
    assert exc.value.errors  # list of messages


def test_dangling_parent_rejected():
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["steps"][1]["parent_id"] = "missing-parent"
    with pytest.raises(TraceValidationError) as exc:
        ingest_trace_dict(bad)
    assert any("missing parent" in e for e in exc.value.errors)


def test_duplicate_step_ids_rejected():
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["steps"][1]["id"] = "s1"
    with pytest.raises(TraceValidationError):
        ingest_trace_dict(bad)


def test_size_limit_bytes():
    with pytest.raises(TraceSizeLimitExceeded):
        enforce_limits({"steps": [{"x": "y" * 100}]}, max_bytes=10)


def test_size_limit_steps():
    raw = {"steps": [{"kind": "note"}] * 10}
    with pytest.raises(TraceSizeLimitExceeded):
        enforce_limits(raw, max_steps=3)


def test_secret_redaction_in_values():
    text = "my api_key is sk-abc123def456ghi789jkl and password=supersecret99"
    out = redact_text(text)
    assert "sk-abc123def456ghi789jkl" not in out
    assert "supersecret99" not in out


def test_secret_redaction_in_keys():
    data = {"api_key": "value", "nested": {"Authorization": "Bearer abcdefghijkl", "ok": 1}}
    out = redact_value(data)
    assert out["api_key"] == "***REDACTED***"
    assert out["nested"]["Authorization"] == "***REDACTED***"
    assert out["nested"]["ok"] == 1


def test_redaction_disabled():
    data = {"api_key": "value"}
    assert redact_value(data, enabled=False) == data


def test_redacted_secrets_never_enter_trace():
    raw = json.loads(json.dumps(VALID_TRACE))
    raw["steps"][0]["tool_call"]["arguments"] = {"api_key": "sk-abcdefghijklmnop123456"}
    t = ingest_trace_dict(raw)
    dumped = t.model_dump_json()
    assert "sk-abcdefghijklmnop123456" not in dumped


def test_load_trace_from_file(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(VALID_TRACE), encoding="utf-8")
    t = load_trace(p)
    assert t.metadata.run_id == "r-1"


def test_load_trace_missing_file():
    with pytest.raises(TraceValidationError):
        load_trace("/no/such/file.json")


def test_load_trace_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(TraceValidationError) as exc:
        load_trace(p)
    assert "Invalid JSON" in str(exc.value)


def test_validate_trace_file_reports_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"metadata": {"run_id": 1}, "steps": [{"kind": "bogus"}]}), encoding="utf-8")
    errs = validate_trace_file(p)
    assert errs


def test_validate_trace_file_valid(tmp_path):
    p = tmp_path / "good.json"
    p.write_text(json.dumps(VALID_TRACE), encoding="utf-8")
    assert validate_trace_file(p) == []


def test_load_trace_bundle(tmp_path):
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps({
        "traces": [VALID_TRACE],
        "expected_findings": [{"category": "repeated_actions", "step_ids": ["s1"]}],
        "description": "test bundle",
    }), encoding="utf-8")
    bundle = load_trace_bundle(p)
    assert len(bundle.traces) == 1
    assert bundle.expected_findings[0]["category"] == "repeated_actions"


def test_openinference_normalization():
    raw = {
        "trace_id": "trace-oi-1",
        "metadata": {"agent_name": "oi-agent", "status": "success"},
        "spans": [
            {
                "span_id": "span1",
                "name": "db_get",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T00:00:01Z",
                "attributes": {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "db_get",
                    "tool.parameters": '{"record_id": "u-1"}',
                    "tool.result": '{"plan": "free"}',
                },
            },
            {
                "span_id": "span2",
                "parent_id": "span1",
                "name": "llm-call",
                "start_time": "2026-01-01T00:00:01Z",
                "attributes": {"openinference.span.kind": "LLM", "llm.input_messages": "hello"},
            },
        ],
    }
    t = ingest_trace_dict(raw)
    assert t.source_format == "openinference"
    assert t.steps[0].kind == StepKind.TOOL_CALL
    assert t.steps[0].tool_call is not None
    assert t.steps[0].tool_call.arguments == {"record_id": "u-1"}
    assert t.steps[1].parent_id == t.steps[0].id


def test_openinference_otlp_shape():
    raw = {
        "resourceSpans": [{"scopeSpans": [{"spans": [
            {"spanId": "s1", "name": "call", "attributes": [{"key": "openinference.span.kind", "value": {"stringValue": "CHAIN"}}]}
        ]}]}]
    }
    t = ingest_trace_dict(raw)
    assert t.steps[0].kind == StepKind.PLANNING


def test_normalize_preserves_missing_values():
    raw = {"spans": [{"span_id": "x", "name": "n", "attributes": {"openinference.span.kind": "LLM"}}]}
    out = normalize_openinference(raw)
    step = out["steps"][0]
    assert step["input"] is None
    assert step["output"] is None
