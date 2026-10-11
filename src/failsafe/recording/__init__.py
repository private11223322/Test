"""Trace recorder: instruments a working agent so its execution can be
recorded as a Failsafe trace.

The recorder is intentionally lightweight: a context-managed object that
collects steps (LLM turns, tool calls, tool results) and writes a valid
Failsafe trace JSON file.  It can wrap tool functions with fault injection.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from failsafe.ingestion import redact_value
from failsafe.schemas import (
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


class TraceRecorder:
    """Records agent execution steps into a :class:`Trace`."""

    def __init__(
        self,
        agent_name: str = "agent",
        agent_version: str | None = None,
        task: TaskSpec | None = None,
        run_id: str | None = None,
        seed: int | None = None,
        redact: bool = True,
    ) -> None:
        self.redact = redact
        meta_kwargs: dict[str, Any] = {
            "agent_name": agent_name,
            "agent_version": agent_version,
            "seed": seed,
        }
        if run_id is not None:
            meta_kwargs["run_id"] = run_id
        if task is not None:
            meta_kwargs["task_id"] = task.id
            meta_kwargs["task_name"] = task.name
        self.metadata = RunMetadata(**meta_kwargs)
        self.trace = Trace(metadata=self.metadata, task=task)
        self._open_parents: list[str] = []
        self._counter = 0

    # -- step helpers -------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self.metadata.run_id}-{self._counter:04d}"

    def add_step(
        self,
        kind: StepKind,
        name: str,
        *,
        status: StepStatus = StepStatus.OK,
        input: Any = None,
        output: Any = None,
        parent_id: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_call: ToolCall | None = None,
        tool_result: ToolResult | None = None,
    ) -> Step:
        step = Step(
            id=self._next_id("step"),
            kind=kind,
            name=name,
            status=status,
            parent_id=parent_id or (self._open_parents[-1] if self._open_parents else None),
            timestamp=datetime.now(UTC),
            input=redact_value(input, self.redact),
            output=redact_value(output, self.redact),
            error_type=error_type,
            error_message=error_message,
            metadata=metadata or {},
            tool_call=tool_call,
            tool_result=tool_result,
        )
        self.trace.steps.append(step)
        return step

    @contextmanager
    def span(self, kind: StepKind, name: str, **kwargs: Any):
        step = self.add_step(kind, name, **kwargs)
        self._open_parents.append(step.id)
        start = time.monotonic()
        try:
            yield step
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            step.status = StepStatus.ERROR
            step.error_type = type(exc).__name__
            step.error_message = str(exc)
            raise
        finally:
            step.end_timestamp = datetime.now(UTC)
            step.metadata.setdefault("duration_ms", (time.monotonic() - start) * 1000)
            self._open_parents.pop()

    def record_llm(self, name: str, prompt: str, response: str, **kwargs: Any) -> Step:
        return self.add_step(
            StepKind.LLM,
            name,
            input={"prompt": prompt},
            output={"response": response},
            **kwargs,
        )

    def record_tool_call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        tc = ToolCall(name=name, arguments=redact_value(arguments, self.redact))
        step = self.add_step(StepKind.TOOL_CALL, name, input=arguments, tool_call=tc)
        tc.step_id = step.id
        return tc

    def record_tool_result(
        self,
        tool_call: ToolCall,
        output: Any = None,
        *,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
        duration_ms: float | None = None,
    ) -> ToolResult:
        tr = ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=success,
            output=redact_value(output, self.redact),
            error_type=error_type,
            error_message=error_message,
            duration_ms=duration_ms,
        )
        step = self.add_step(
            StepKind.TOOL_RESULT,
            tool_call.name,
            status=StepStatus.OK if success else StepStatus.ERROR,
            output=tr.output,
            tool_result=tr,
            error_type=error_type,
            error_message=error_message,
        )
        tr.step_id = step.id
        return tr

    def record_response(self, text: str, claimed_success: bool | None = None) -> Step:
        meta: dict[str, Any] = {}
        if claimed_success is not None:
            meta["claimed_success"] = claimed_success
        return self.add_step(StepKind.RESPONSE, "final_response", output=text, metadata=meta)

    def record_error(self, message: str, error_type: str = "agent_error") -> Step:
        return self.add_step(
            StepKind.ERROR, "error", status=StepStatus.ERROR,
            error_type=error_type, error_message=message,
        )

    def record_fault_injection(self, record: Any) -> None:
        record.run_id = self.metadata.run_id
        self.trace.fault_injections.append(record)

    # -- lifecycle ----------------------------------------------------
    def finish(self, status: RunStatus = RunStatus.SUCCESS, final_response: str | None = None, error: str | None = None) -> Trace:
        self.metadata.status = status
        self.metadata.ended_at = datetime.now(UTC)
        if final_response is not None:
            self.metadata.final_response = redact_value(final_response, self.redact)
        if error is not None:
            self.metadata.error = error
        # Pydantic copies nested models on construction: reattach live metadata
        self.trace.metadata = self.metadata
        return self.trace

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.trace.model_dump(mode="json")
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return p
