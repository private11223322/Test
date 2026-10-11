"""Trace loading and validation.

Loads JSON trace fixtures in the Failsafe native format or the
OpenInference/OpenTelemetry-GenAI span format and validates them into the
normalized internal :class:`~failsafe.schemas.Trace` model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from failsafe.ingestion import enforce_limits, redact_value
from failsafe.schemas import Trace, TraceBundle


class TraceValidationError(Exception):
    """Raised when a trace cannot be validated.  Carries useful messages."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        out.append(f"{loc}: {err.get('msg', 'invalid')}")
    return out


def load_raw(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        raise TraceValidationError(f"File not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceValidationError(
            f"Invalid JSON in {p}: {exc.msg} at line {exc.lineno} column {exc.colno}",
            errors=[str(exc)],
        ) from exc


def detect_format(raw: Any) -> str:
    if isinstance(raw, dict):
        if "metadata" in raw and "steps" in raw:
            return "failsafe"
        if "spans" in raw or "resourceSpans" in raw:
            return "openinference"
        if raw.get("trace_id") and (raw.get("spans") or raw.get("steps")):
            return "openinference"
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        first = raw[0]
        if "span_id" in first or "attributes" in first:
            return "openinference"
        if "kind" in first and "id" in first:
            return "failsafe-steps"
    return "unknown"


def ingest_trace_dict(
    raw: dict[str, Any],
    *,
    redact: bool = True,
    max_bytes: int = 5_000_000,
    max_steps: int = 5_000,
    source_format: str = "failsafe",
) -> Trace:
    """Validate and normalize a raw trace dict into a :class:`Trace`."""
    enforce_limits(raw, max_bytes=max_bytes, max_steps=max_steps)
    cleaned = redact_value(raw, enabled=redact)

    fmt = detect_format(cleaned)
    if fmt == "openinference":
        from failsafe.ingestion.normalize import normalize_openinference

        cleaned = normalize_openinference(cleaned)

    try:
        trace = Trace.model_validate(cleaned)
    except ValidationError as exc:
        raise TraceValidationError(
            "Trace failed schema validation: " + "; ".join(_format_pydantic_errors(exc)[:10]),
            errors=_format_pydantic_errors(exc),
        ) from exc

    trace.source_format = source_format if fmt != "openinference" else "openinference"
    _cross_validate(trace)
    return trace


def _cross_validate(trace: Trace) -> None:
    """Semantic validation beyond the schema: dangling references etc."""
    errors: list[str] = []
    step_ids = {s.id for s in trace.steps}
    if len(step_ids) != len(trace.steps):
        errors.append("Duplicate step IDs detected.")
    for s in trace.steps:
        if s.parent_id is not None and s.parent_id not in step_ids:
            errors.append(f"Step {s.id} references missing parent {s.parent_id}.")
        if s.tool_call is not None and s.tool_call.step_id is None:
            s.tool_call.step_id = s.id
        if s.tool_result is not None and s.tool_result.step_id is None:
            s.tool_result.step_id = s.id
    for fi in trace.fault_injections:
        if fi.step_id is not None and fi.step_id not in step_ids:
            errors.append(f"Fault injection {fi.id} references missing step {fi.step_id}.")
    if errors:
        raise TraceValidationError(
            "Trace failed semantic validation: " + "; ".join(errors[:10]),
            errors=errors,
        )


def load_trace(
    path: str | Path,
    *,
    redact: bool = True,
    max_bytes: int = 5_000_000,
    max_steps: int = 5_000,
) -> Trace:
    raw = load_raw(path)
    if not isinstance(raw, dict):
        raise TraceValidationError("Trace file must contain a JSON object at the top level.")
    return ingest_trace_dict(
        raw, redact=redact, max_bytes=max_bytes, max_steps=max_steps, source_format=str(path)
    )


def load_trace_bundle(
    path: str | Path,
    *,
    redact: bool = True,
    max_bytes: int = 5_000_000,
    max_steps: int = 5_000,
) -> TraceBundle:
    """Load either a single trace or a bundle ``{"traces": [...]}``."""
    raw = load_raw(path)
    if isinstance(raw, dict) and isinstance(raw.get("traces"), list):
        traces = [
            ingest_trace_dict(t, redact=redact, max_bytes=max_bytes, max_steps=max_steps)
            for t in raw["traces"]
        ]
        return TraceBundle(
            traces=traces,
            expected_findings=list(raw.get("expected_findings", [])),
            description=str(raw.get("description", "")),
        )
    if isinstance(raw, dict):
        return TraceBundle(traces=[ingest_trace_dict(raw, redact=redact, max_bytes=max_bytes, max_steps=max_steps)])
    raise TraceValidationError("Unsupported trace bundle structure.")


def validate_trace_file(path: str | Path) -> list[str]:
    """Return a list of validation errors (empty when valid)."""
    try:
        load_trace(path)
        return []
    except TraceValidationError as exc:
        return exc.errors or [str(exc)]
