"""Normalize OpenInference / OpenTelemetry-GenAI style spans into the
Failsafe internal trace model.

Supported input shapes:

1. ``{"spans": [<span>, ...], "metadata": {...}?}``
2. A bare list of spans (wrapped by the caller).
3. OTLP-ish ``{"resourceSpans": [{"scopeSpans": [{"spans": [...]}]}]}``

Each span may carry OpenInference attributes such as
``openinference.span.kind`` (``LLM``, ``TOOL``, ``RETRIEVER``, ``CHAIN``),
``tool.name``, ``tool.parameters``, ``tool.result``, ``llm.input_messages``,
``llm.output_messages``, ``llm.token_count`` etc.  Unknown attributes are
preserved under ``metadata.attributes``.  Missing values are preserved as
``None`` and never invented.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from failsafe.schemas import (
    RunStatus,
    StepKind,
    StepStatus,
    ToolCall,
    ToolResult,
)

_KIND_MAP = {
    "LLM": StepKind.LLM,
    "TOOL": StepKind.TOOL_CALL,
    "RETRIEVER": StepKind.RETRIEVAL,
    "CHAIN": StepKind.PLANNING,
    "AGENT": StepKind.PLANNING,
    "EMBEDDING": StepKind.RETRIEVAL,
    "PARSER": StepKind.OTHER,
    "GUARDRAIL": StepKind.OTHER,
}


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # nanosecond epoch (OTLP) or second epoch
        if value > 1e12:
            return datetime.fromtimestamp(value / 1e9, tz=UTC)
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            return None
    return None


def _attr(span: dict[str, Any], name: str) -> Any:
    """Fetch an attribute from either flat span dict or OTLP attributes list."""
    attrs = span.get("attributes")
    if isinstance(attrs, dict):
        return attrs.get(name)
    if isinstance(attrs, list):
        for item in attrs:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if key != name:
                continue
            val = item.get("value", {})
            if isinstance(val, dict):
                for v in val.values():
                    return v
            return val
    # also allow flat keys directly on span
    return span.get(name)


def _jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _iter_spans(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw.get("spans"), list):
        return [s for s in raw["spans"] if isinstance(s, dict)]
    if isinstance(raw.get("resourceSpans"), list):
        spans: list[dict[str, Any]] = []
        for rs in raw["resourceSpans"]:
            for ss in rs.get("scopeSpans", []) or rs.get("instrumentationLibrarySpans", []) or []:
                for s in ss.get("spans", []) or []:
                    if isinstance(s, dict):
                        spans.append(s)
        return spans
    return []


def normalize_openinference(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a span-based trace into the Failsafe native dict shape."""
    spans = _iter_spans(raw)
    # sort by start time when available
    spans.sort(key=lambda s: (_parse_ts(s.get("start_time") or s.get("startTime")) or datetime.min.replace(tzinfo=UTC)))

    steps: list[dict[str, Any]] = []
    span_id_to_step: dict[str, str] = {}

    meta_raw = raw.get("metadata") or {}
    run_id = meta_raw.get("run_id") or raw.get("trace_id") or raw.get("traceId")
    status = meta_raw.get("status") or "unknown"

    for idx, span in enumerate(spans):
        span_id = str(span.get("span_id") or span.get("spanId") or f"span-{idx}")
        parent = span.get("parent_id") or span.get("parentSpanId")
        kind_raw = _attr(span, "openinference.span.kind") or span.get("kind") or "OTHER"
        kind = _KIND_MAP.get(str(kind_raw).upper(), StepKind.OTHER)

        status_raw = span.get("status") or {}
        if isinstance(status_raw, dict):
            status_code = str(status_raw.get("code") or status_raw.get("status_code") or "").upper()
            status_message = status_raw.get("message")
        else:
            status_code = str(status_raw).upper()
            status_message = span.get("status_message")
        if status_code in ("ERROR", "2", "STATUS_CODE_ERROR"):
            step_status = StepStatus.ERROR
        elif status_code in ("TIMEOUT",):
            step_status = StepStatus.TIMEOUT
        else:
            step_status = StepStatus.OK

        start = _parse_ts(span.get("start_time") or span.get("startTime")) or datetime.now(UTC)
        end = _parse_ts(span.get("end_time") or span.get("endTime"))

        name = span.get("name") or _attr(span, "tool.name") or _attr(span, "llm.model_name") or f"step-{idx}"

        tool_call = None
        tool_result = None
        if kind in (StepKind.TOOL_CALL,):
            tc_name = _attr(span, "tool.name") or span.get("name") or "unknown-tool"
            tc_args = _jsonish(_attr(span, "tool.parameters") or _attr(span, "tool.arguments") or {})
            if not isinstance(tc_args, dict):
                tc_args = {"_raw": tc_args}
            tool_call = ToolCall(name=str(tc_name), arguments=tc_args)
            result_val = _attr(span, "tool.result")
            if result_val is not None or step_status == StepStatus.OK:
                tool_result = ToolResult(
                    tool_call_id=tool_call.id,
                    name=str(tc_name),
                    success=step_status == StepStatus.OK,
                    output=_jsonish(result_val),
                    error_type=None if step_status == StepStatus.OK else "tool_error",
                    error_message=status_message,
                )

        input_val = _jsonish(_attr(span, "llm.input_messages") or _attr(span, "input.value") or span.get("input"))
        output_val = _jsonish(_attr(span, "llm.output_messages") or _attr(span, "output.value") or span.get("output"))

        step: dict[str, Any] = {
            "id": f"step-{span_id}" if not span_id.startswith("step-") else span_id,
            "kind": kind.value,
            "name": str(name),
            "status": step_status.value,
            "timestamp": start.isoformat(),
            "end_timestamp": end.isoformat() if end else None,
            "input": input_val,
            "output": output_val,
            "error_type": None if step_status == StepStatus.OK else (span.get("error_type") or "span_error"),
            "error_message": status_message,
            "metadata": {"attributes": {}, "span_id": span_id},
        }
        if tool_call is not None:
            step["tool_call"] = tool_call.model_dump(mode="json")
        if tool_result is not None:
            step["tool_result"] = tool_result.model_dump(mode="json")

        # preserve unknown attributes
        attrs = span.get("attributes")
        known = {
            "openinference.span.kind", "tool.name", "tool.parameters", "tool.arguments",
            "tool.result", "llm.input_messages", "llm.output_messages", "llm.model_name",
            "input.value", "output.value",
        }
        if isinstance(attrs, dict):
            step["metadata"]["attributes"] = {k: v for k, v in attrs.items() if k not in known}

        span_id_to_step[span_id] = step["id"]
        steps.append(step)

    # fix parent references
    for span, step in zip(spans, steps, strict=False):
        parent = span.get("parent_id") or span.get("parentSpanId")
        if parent is not None:
            step["parent_id"] = span_id_to_step.get(str(parent))

    metadata: dict[str, Any] = {
        "run_id": run_id or f"run-ot-{idx if spans else 0}",
        "agent_name": meta_raw.get("agent_name") or "openinference-agent",
        "status": _map_run_status(status),
    }
    if meta_raw.get("agent_version"):
        metadata["agent_version"] = meta_raw["agent_version"]
    if meta_raw.get("task_id"):
        metadata["task_id"] = meta_raw["task_id"]

    out: dict[str, Any] = {
        "metadata": metadata,
        "steps": steps,
        "source_format": "openinference",
    }
    if raw.get("task"):
        out["task"] = raw["task"]
    return out


def _map_run_status(status: Any) -> str:
    s = str(status).lower()
    if s in ("success", "ok", "completed", "succeeded"):
        return RunStatus.SUCCESS.value
    if s in ("failure", "failed"):
        return RunStatus.FAILURE.value
    if s in ("error", "crash", "crashed"):
        return RunStatus.ERROR.value
    return RunStatus.UNKNOWN.value
